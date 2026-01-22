"""
COPYRIGHT NOTICE

© 2025 Anonymous Author(s). All rights reserved.

This software and its associated documentation files (the "Software") are owned by Anonymous Author(s).
The Software is protected by copyright laws and international copyright treaties, as well as other intellectual property laws and treaties.
Unauthorized use, reproduction, modification, or distribution of the Software is strictly prohibited.

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/.

Under the terms of this license, you are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material.

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- Non-Commercial: You may not use the material for commercial purposes.
- ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

PyTorch Version - Converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from tqdm import tqdm


def gaussian_blur_2d(image, kernel_size, sigma):
    """Apply Gaussian blur to a batch of images using reflection padding."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma), float(sigma))

    def gaussian_kernel_1d(size, sigma):
        x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        return kernel / kernel.sum()

    kernel_y = gaussian_kernel_1d(kernel_size[0], sigma[0]).view(-1, 1)
    kernel_x = gaussian_kernel_1d(kernel_size[1], sigma[1]).view(1, -1)
    kernel_2d = kernel_y @ kernel_x
    kernel_2d = kernel_2d.view(1, 1, kernel_size[0], kernel_size[1])

    device = image.device
    kernel_2d = kernel_2d.to(device)

    channels = image.shape[1]
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)

    # Use reflection padding instead of zero padding to avoid dark borders
    pad_h = kernel_size[0] // 2
    pad_w = kernel_size[1] // 2
    image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    return F.conv2d(image_padded, kernel_2d, padding=0, groups=channels)


class ResidualModule(nn.Module):
    def __init__(self, filters=256, kernel_size=3, stride=1, padding='same', activation="relu", dilation_rate=1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation_name = activation

        if padding == 'same':
            self.pad_val = (kernel_size + (kernel_size - 1) * (dilation_rate - 1) - 1) // 2
        else:
            self.pad_val = 0

        self.conv0 = None
        self.bn0 = None

        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters // 2, kernel_size=1, stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(filters // 2)
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=kernel_size,
                               stride=stride, padding=self.pad_val, dilation=dilation_rate)

        self.bn3 = nn.BatchNorm2d(filters // 2)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0)

        self._initialized = False

    def _get_activation(self):
        if self.activation_name.lower() == "relu":
            return F.relu
        elif self.activation_name.lower() == "gelu":
            return F.gelu
        elif self.activation_name.lower() == "linear":
            return lambda x: x
        else:
            return F.relu

    def forward(self, x):
        activation = self._get_activation()

        if not self._initialized or self.conv0 is None:
            in_channels = x.shape[1]
            if in_channels != self.filters:
                self.conv0 = nn.Conv2d(in_channels, self.filters, kernel_size=1, stride=1, padding=0).to(x.device)
                self.bn0 = nn.BatchNorm2d(self.filters).to(x.device)
            self._initialized = True

        if x.shape[1] == self.filters:
            identity = x
        else:
            identity = self.conv0(activation(self.bn0(x)))

        out = self.conv1(activation(self.bn1(identity)))
        out = self.conv2(activation(self.bn2(out)))
        out = self.conv3(activation(self.bn3(out)))

        out = out + identity

        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class ConfMSELoss(nn.Module):
    """
    SCAN Loss Function based on Information Bottleneck Theory.

    Combines Confidence Loss (Eq. 8) and Reconstruction Loss (Eq. 9) from the paper.

    Args:
        alpha: Controls the target area size of self-confidence map (Eq. 5: Ac = 1/(1+α))
        lambd: Flexibility constant for confidence loss (default: 0.1)
    """
    def __init__(self, alpha=4.0, lambd=0.1):
        super().__init__()
        self.alpha = alpha
        self.lambd = lambd
        self.conf_lowest_point = 1 / (1 + self.alpha)  # Ac from Eq. 5

    def _stretching_sine_function(self, x, stretch_alpha=0.15, cycle_factor=8.0):
        """
        Stretching sine activation function (Eq. 4 from paper).

        Ĉ_i = (Ŷc/|Ŷc| * sin(2π|Ŷc|/(8+0.15|Ŷc|)) + 1) / 2

        Args:
            x: Input tensor (raw confidence output)
            stretch_alpha: Stretching factor (0.15 in paper)
            cycle_factor: Base cycle factor (8.0 in paper)
        """
        sign = x / torch.clamp(torch.abs(x), min=1e-13)
        denominator = torch.clamp(cycle_factor + stretch_alpha * torch.abs(x), min=1e-13)
        return (sign * torch.sin((2 * np.pi * torch.abs(x)) / denominator) + 1.) / 2.

    def forward(self, y_pred, y_true):
        """
        Compute total loss = Loss_c + Loss_r (Eq. 10)

        Args:
            y_pred: Decoder output [B, 4, H, W] - first channel is confidence, rest is RGB
            y_true: Target blurred image [B, 3, H, W] normalized to [-1, 1]
        """
        # Split prediction into confidence map and reconstructed image (Eq. 3)
        conf = self._stretching_sine_function(y_pred[:, 0:1, :, :])  # Ĉ
        pred = y_pred[:, 1:, :, :]  # Ŷr

        # Squared error (matching TensorFlow: tf.maximum(tf.abs(...), 1e-13)**2)
        sq_err = torch.clamp(torch.abs(y_true - pred), min=1e-13) ** 2

        # Compute ω (Eq. 6): ω = |Ĉμ - Ac|² / (Ĉμ * (1 - Ĉμ))
        # Note: TensorFlow uses tf.abs(omega - self.conf_lowest_point)
        conf_mean = torch.clamp(conf.mean(dim=[-3, -2, -1], keepdim=True), min=1e-13)  # Ĉμ
        omega = torch.abs(conf_mean - self.conf_lowest_point) ** 2 / torch.clamp(conf_mean * (1 - conf_mean), min=1e-13)

        # Confidence Loss (Eq. 8): Loss_c = (1+ω) * (||Ỹ - Ŷr||² + λ) - λ
        mse = sq_err.mean(dim=[-3, -2, -1])
        loss_conf = (1 + omega.squeeze()) * (mse + self.lambd) - self.lambd

        # Reconstruction Loss (Eq. 9): Loss_r = mean(αĈ * err² + (1-Ĉ) * err²)
        loss_recovery = (self.alpha * conf * sq_err + (1 - conf) * sq_err).mean(dim=[-3, -2, -1])

        # Total Loss (Eq. 10)
        return (loss_conf + loss_recovery).mean()


class ConfMAE_Metric:
    """Confident MAE Metric - measures MAE weighted by confidence."""
    def __init__(self):
        self.reset()

    def _stretching_sine_function(self, x, stretch_alpha=0.15, cycle_factor=8.0):
        """Stretching sine activation (Eq. 4)."""
        sign = x / torch.clamp(torch.abs(x), min=1e-13)
        denominator = torch.clamp(cycle_factor + stretch_alpha * torch.abs(x), min=1e-13)
        return (sign * torch.sin((2 * np.pi * torch.abs(x)) / denominator) + 1.) / 2.

    def update(self, y_true, y_pred):
        conf = self._stretching_sine_function(y_pred[:, 0:1, :, :])
        pred = y_pred[:, 1:, :, :]

        value = conf * torch.abs(y_true - pred)
        value = value.sum() / torch.clamp(conf.sum(), min=1e-13)

        self.value += value.item()
        self.cnt += 1

    def compute(self):
        return self.value / max(self.cnt, 1e-17)

    def reset(self):
        self.value = 0.0
        self.cnt = 0


class NoConfMAE_Metric:
    """Not-Confident MAE Metric - measures MAE weighted by (1 - confidence)."""
    def __init__(self):
        self.reset()

    def _stretching_sine_function(self, x, stretch_alpha=0.15, cycle_factor=8.0):
        """Stretching sine activation (Eq. 4)."""
        sign = x / torch.clamp(torch.abs(x), min=1e-13)
        denominator = torch.clamp(cycle_factor + stretch_alpha * torch.abs(x), min=1e-13)
        return (sign * torch.sin((2 * np.pi * torch.abs(x)) / denominator) + 1.) / 2.

    def update(self, y_true, y_pred):
        conf = self._stretching_sine_function(y_pred[:, 0:1, :, :])
        pred = y_pred[:, 1:, :, :]

        value = (1 - conf) * torch.abs(y_true - pred)
        value = value.sum() / torch.clamp((1 - conf).sum(), min=1e-13)

        self.value += value.item()
        self.cnt += 1

    def compute(self):
        return self.value / max(self.cnt, 1e-17)

    def reset(self):
        self.value = 0.0
        self.cnt = 0


class ConvolutionalDecoder(nn.Module):
    """
    Convolutional Decoder following the original SCAN implementation.

    Structure (for level 3, feature_size=7):
    - Input projection to initial channels
    - Level-based upsampling blocks with skip connections
    - Final upsampling to 128 -> 64 channels
    - Output: 4 channels (1 confidence + 3 RGB)
    """
    def __init__(self, input_shape, img_size=(224, 224), ch_per_lv=[192, 256, 384, 512]):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]
        feature_size = input_shape[1]

        # Determine level based on feature size (7->3, 14->2, 28->1, 56->0)
        # level = number of upsampling steps needed before reaching 56x56
        self.level = int(np.log2(img_size[0] / feature_size)) - 2
        self.level = max(0, min(3, self.level))

        # Initial projection: input channels -> ch_per_lv[level]
        initial_ch = ch_per_lv[self.level]
        self.input_proj = nn.Conv2d(in_channels, initial_ch, kernel_size=1, stride=1, padding=0)
        self.initial_res = ResidualModule(initial_ch, 1, 1, activation="relu", padding="same")

        # Level-based blocks (each with skip connection)
        self.level_blocks = nn.ModuleList()
        self.level_upsamples = nn.ModuleList()

        # Build blocks for each level (from current level down to 0)
        for lv in range(self.level, -1, -1):
            ch = ch_per_lv[lv]
            self.level_blocks.append(nn.ModuleList([
                ResidualModule(ch, 3, 1, activation="relu", padding="same"),
                ResidualModule(ch, 3, 1, activation="relu", padding="same"),
            ]))
            if lv > 0:
                next_ch = ch_per_lv[lv - 1]
                self.level_upsamples.append(nn.Sequential(
                    nn.ConvTranspose2d(ch, next_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU()
                ))

        # Final blocks: 192 -> 128 -> 64
        self.final_up1 = nn.Sequential(
            nn.ConvTranspose2d(ch_per_lv[0], 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.final_block1 = nn.ModuleList([
            ResidualModule(128, 3, 1, activation="relu", padding="same"),
            ResidualModule(128, 3, 1, activation="relu", padding="same"),
        ])

        self.final_up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.final_block2 = nn.ModuleList([
            ResidualModule(64, 3, 1, activation="relu", padding="same"),
            ResidualModule(64, 3, 1, activation="relu", padding="same"),
        ])

        # Output layers
        self.final_res = ResidualModule(64, 1, 1, activation="relu", padding="same")
        self.output_conv = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.initial_res(x)
        _x = x

        # Level-based blocks with skip connections
        for i, (block_list, upsample) in enumerate(zip(self.level_blocks[:-1], self.level_upsamples)):
            for block in block_list:
                x = block(x)
            x = x + _x  # Skip connection
            x = upsample(x)
            _x = x

        # Last level block (no upsample after)
        for block in self.level_blocks[-1]:
            x = block(x)
        x = x + _x  # Skip connection

        # Final upsampling blocks with skip connections
        x = self.final_up1(x)
        _x = x
        for block in self.final_block1:
            x = block(x)
        x = x + _x  # Skip connection

        x = self.final_up2(x)
        _x = x
        for block in self.final_block2:
            x = block(x)
        x = x + _x  # Skip connection

        # Output
        x = self.final_res(x)
        x = self.output_conv(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, input_shape, img_size=(224, 224)):
        super().__init__()

        self.input_shape = input_shape
        seq_len = input_shape[0]
        emb_dim = input_shape[1]

        grid_size = int(seq_len ** 0.5)
        self.grid_size = grid_size
        self.emb_dim = emb_dim
        repeat_cnt = int(np.log(img_size[0] / grid_size) / np.log(2))

        self.transformer_blocks = nn.Sequential(
            TransformerBlock(emb_dim, 12),
            TransformerBlock(emb_dim, 12),
            TransformerBlock(emb_dim, 12),
            TransformerBlock(emb_dim, 12),
        )

        self.has_class_token = (grid_size ** 2 != seq_len)

        self.upsampling = nn.ModuleList()
        in_ch = emb_dim
        for i in range(repeat_cnt):
            out_ch = 32 * 2 ** (repeat_cnt - 1 - i)
            self.upsampling.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                ResidualModule(out_ch, 3, 1, activation="relu", padding="same"),
                ResidualModule(out_ch, 3, 1, activation="relu", padding="same"),
            ))
            in_ch = out_ch

        self.output_conv = nn.Conv2d(in_ch, 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.transformer_blocks(x)

        if self.has_class_token:
            x = x[:, 1:]

        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch_size, self.emb_dim, self.grid_size, self.grid_size)

        for block in self.upsampling:
            x = block(x)

        x = self.output_conv(x)

        return x


class SCAN:
    '''
    SCAN Library (PyTorch Version)

    The SCAN library is designed to extract visual explanations from deep learning models.
    By specifying the model and layer to be analyzed, it learns to analyze visual explanations
    using the feature map extracted from the specified layer. Once trained, it can output
    visual explanations for all input images.

    Steps to use this library:
    1. Specify the target model and layer. It works with most models and layers.
    2. Specify the training dataset and validation dataset (optional).
    3. Create the decoder model (convolutional or transformer networks).
    4. Compile the decoder (specify Alpha value, optimizer, learning rate, and metrics).
    5. Train the decoder using fit or train function.
    6. SCAN inference - input an image to get visual explanation.
    '''

    def __init__(self, target_model, target_layer, image_size=(224, 224), use_gradient_mask=True,
                 decoder_model=None, device=None, num_classes=1000):
        '''
        Parameters:
            target_model: PyTorch model (nn.Module)
                The model object to be analyzed.
            target_layer: str or nn.Module or int
                The layer to extract features from. Can be layer name, module reference, or index.
            image_size: tuple
                Input image size (H, W). Default is (224, 224).
            use_gradient_mask: bool
                Whether to use gradient masking during training. Default is True.
            decoder_model: nn.Module or None
                If an existing decoder model is available, it can be provided here.
            device: str or torch.device
                Device to use ('cuda' or 'cpu'). Auto-detected if None.
            num_classes: int
                Number of classes for the target model. Default is 1000 (ImageNet).
        '''

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = target_model.to(self.device)
        self.model.eval()
        self.decoder = decoder_model
        self.target_layer = target_layer
        self.use_gradient_mask = use_gradient_mask
        self.IMG_SIZE = image_size
        self.num_classes = num_classes

        self.valid_dataset = None
        self.preprocess = lambda x: x

        self._features = None
        self._hook_handle = None
        self._setup_hooks()

        self.optimizer = None
        self.criterion = None
        self.metrics = []
        # Default augmentation range: randomly sample P from 70-100% during training (paper Section IV-B-1)
        self.use_augmentation = (70, 100)

    def _setup_hooks(self):
        """Setup forward hooks to extract features from target layer."""
        def hook_fn(module, input, output):
            self._features = output

        if isinstance(self.target_layer, str):
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    self._hook_handle = module.register_forward_hook(hook_fn)
                    return
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        elif isinstance(self.target_layer, nn.Module):
            self._hook_handle = self.target_layer.register_forward_hook(hook_fn)
        elif isinstance(self.target_layer, int):
            modules = list(self.model.modules())[1:]
            if self.target_layer < len(modules):
                self._hook_handle = modules[self.target_layer].register_forward_hook(hook_fn)
            else:
                raise ValueError(f"Layer index {self.target_layer} out of range")

    def _stretching_sine_function(self, x, stretch_alpha=0.15, cycle_factor=8.0):
        """
        Stretching sine activation function (Eq. 4 from paper).

        Ĉ_i = (Ŷc/|Ŷc| * sin(2π|Ŷc|/(8+0.15|Ŷc|)) + 1) / 2
        """
        sign = x / torch.clamp(torch.abs(x), min=1e-13)
        denominator = torch.clamp(cycle_factor + stretch_alpha * torch.abs(x), min=1e-13)
        return (sign * torch.sin((2 * np.pi * torch.abs(x)) / denominator) + 1.) / 2.

    def _nanpercentile(self, x, percentage_range=(70, 100)):
        q = percentage_range[0] + np.random.rand() * (percentage_range[1] - percentage_range[0])
        x_np = x.detach().cpu().numpy().flatten()
        x_np = x_np[~np.isnan(x_np)]
        if len(x_np) == 0:
            return 0.0
        return float(np.nanpercentile(x_np, q))

    def _gradient_map(self, X, class_idx=None):
        """Compute gradient map for input images."""
        nobatch = len(X.shape) == 3

        if nobatch:
            X = X.unsqueeze(0)

        X = X.to(self.device)

        if self.use_gradient_mask:
            X_input = X.clone().requires_grad_(True)
            self.model.zero_grad()

            preprocessed = self.preprocess(X_input)
            output = self.model(preprocessed)
            features = self._features

            if features.requires_grad is False:
                features = features.clone().requires_grad_(True)

            if class_idx is None:
                class_idx = output.argmax(dim=-1)
            elif isinstance(class_idx, int):
                # Convert int to tensor
                class_idx = torch.tensor([class_idx], device=self.device)
            elif not isinstance(class_idx, torch.Tensor):
                class_idx = torch.tensor(class_idx, device=self.device)

            one_hot = F.one_hot(class_idx, self.num_classes).float()
            pred_c = (output * one_hot).sum()

            if features.grad_fn is not None:
                gradmap = torch.autograd.grad(pred_c, features, retain_graph=True, create_graph=False)[0]
            else:
                gradmap = torch.ones_like(features)

            gradmap = torch.clamp(gradmap, min=0)

            if nobatch:
                features = features[0]
                gradmap = gradmap[0]

            return features.detach(), gradmap.detach()
        else:
            with torch.no_grad():
                preprocessed = self.preprocess(X)
                _ = self.model(preprocessed)
                features = self._features

            if nobatch:
                features = features[0]

            return features.detach(), torch.ones_like(features)

    def _process_batch(self, images, labels=None, use_augmentation=True):
        """Process a batch of images for decoder training.

        Args:
            images: Input images [B, C, H, W] in [0, 255] range
            labels: Ground truth class labels [B] - IMPORTANT for correct gradient computation
            use_augmentation: Whether to apply percentile augmentation
        """
        images = F.interpolate(images, size=self.IMG_SIZE, mode='bilinear', align_corners=False)

        feature_maps = []
        blurred_images = []

        for i in range(images.shape[0]):
            image = images[i:i+1]
            # Use ground truth label for gradient computation (critical for training!)
            label = labels[i].item() if labels is not None else None
            feature_map, gradient_map = self._gradient_map(image, class_idx=label)

            if feature_map.dim() == 3:
                feature_map = feature_map.unsqueeze(0)
                gradient_map = gradient_map.unsqueeze(0)

            if not use_augmentation or self.use_augmentation is False:
                gmask_map = (gradient_map > 0).float()
            else:
                Qmask = gradient_map[gradient_map > 0]
                if len(Qmask) > 0:
                    Q = self._nanpercentile(Qmask, self.use_augmentation)
                else:
                    Q = 0
                gmask_map = (gradient_map >= Q).float()

            gradient_masked_feature_map = feature_map * gmask_map
            feature_maps.append(gradient_masked_feature_map)

            # Gaussian blur parameters (from original implementation)
            # Given original image size sk×sk and feature map size sf×sf:
            # - blur_sigma = sk / sf
            # - blur_kernel = int(blur_sigma * 2) // 2 * 2 + 1 (ensures odd kernel)
            feature_size = feature_map.shape[-1]  # sf
            img_size = self.IMG_SIZE[0]  # sk
            blur_sigma = img_size / feature_size  # σ = sk/sf (e.g., 224/7 = 32)
            blur_kernel = int(blur_sigma * 2) // 2 * 2 + 1  # ensures odd kernel size

            blurred = gaussian_blur_2d(image, blur_kernel, blur_sigma)
            blurred_images.append(blurred)

        feature_maps = torch.cat(feature_maps, dim=0)
        blurred_images = torch.cat(blurred_images, dim=0)

        blurred_images = blurred_images / 127.5 - 1.

        return feature_maps, blurred_images

    def load_decoder(self, filepath):
        '''Load a trained decoder model file.'''
        self.decoder = torch.load(filepath, map_location=self.device)
        self.decoder.to(self.device)

    def save_decoder(self, filepath):
        '''Saves a trained decoder model to a file.'''
        torch.save(self.decoder, filepath)

    def set_preprocess(self, func):
        '''Sets the preprocessing function used by the target model.'''
        self.preprocess = func
        return self

    def set_dataset(self, dataloader, use_augmentation=(70, 100)):
        '''
        Sets the dataset for training the decoder model.

        Parameters:
            dataloader: torch.utils.data.DataLoader
                The DataLoader containing (image, label) pairs.
                Images should be in (N, C, H, W) format with values in [0, 255].
            use_augmentation: tuple or bool
                Percentile augmentation range for gradient masking during training.
                Default is (70, 100) as specified in paper Section IV-B-1.
                Set to False to disable augmentation.

        Returns:
            self: The instance of the class.
        '''
        self.use_augmentation = use_augmentation
        self.dataset = dataloader
        return self

    def set_validation_dataset(self, dataloader):
        '''
        Sets the dataset for evaluating the decoder model.

        Parameters:
            dataloader: torch.utils.data.DataLoader
                The DataLoader containing (image, label) pairs.

        Returns:
            self: The instance of the class.
        '''
        self.valid_dataset = dataloader
        return self

    def _get_feature_shape(self):
        """Get the shape of features from target layer."""
        dummy_input = torch.randn(1, 3, *self.IMG_SIZE).to(self.device)
        with torch.no_grad():
            self.model(self.preprocess(dummy_input))
            feature_shape = self._features.shape[1:]
        return feature_shape

    def generate_decoder(self, is_Transformer=False, ch_per_lv=[192, 256, 384, 512]):
        '''
        Generates the decoder model based on the specified type.

        Parameters:
            is_Transformer: bool
                If True, generates a transformer-based decoder model.
                If False, generates a convolutional-based decoder model. Default is False.
            ch_per_lv: list
                Channel configuration per level for convolutional decoder.
                Default is [192, 256, 384, 512] following original implementation.
                Index 0 = level 0 (56x56), Index 3 = level 3 (7x7).

        Returns:
            self: The instance of the class with the generated decoder model.
        '''
        feature_shape = self._get_feature_shape()

        if is_Transformer:
            self.decoder = TransformerDecoder(feature_shape, self.IMG_SIZE)
        else:
            self.decoder = ConvolutionalDecoder(feature_shape, self.IMG_SIZE, ch_per_lv)

        self.decoder = self.decoder.to(self.device)
        return self

    def compile(self, loss_alpha=4.0, optimizer_class=None, learning_rate=1e-3,
                metrics=None, scheduler_class=None, scheduler_kwargs=None):
        '''
        Compiles the decoder model with the specified parameters.

        Parameters:
            loss_alpha: float
                The alpha value used in the loss function. Default is 4.0.
            optimizer_class: torch.optim.Optimizer class
                The optimizer class to use for training. Default is Adam.
            learning_rate: float
                The learning rate for the optimizer. Default is 1e-3.
            metrics: list
                A list of metric classes to evaluate during training.
            scheduler_class: torch.optim.lr_scheduler class or None
                The learning rate scheduler class. Default is None.
            scheduler_kwargs: dict or None
                Keyword arguments for the scheduler. Default is None.

        Returns:
            self: The instance of the class with the compiled decoder model.
        '''
        if optimizer_class is None:
            optimizer_class = torch.optim.Adam
        if metrics is None:
            metrics = [ConfMAE_Metric, NoConfMAE_Metric]

        self.criterion = ConfMSELoss(loss_alpha)
        self.optimizer = optimizer_class(self.decoder.parameters(), lr=learning_rate)
        self.metrics = [metric() for metric in metrics]

        # Setup learning rate scheduler
        self.scheduler = None
        if scheduler_class is not None:
            scheduler_kwargs = scheduler_kwargs or {}
            self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

        return self

    def fit(self, epochs=2, verbose=True):
        '''
        Trains the decoder model on the dataset.

        Parameters:
            epochs: int
                The number of epochs to train the model. Default is 2.
            verbose: bool
                Whether to print progress. Default is True.

        Returns:
            history: dict
                Training history containing loss and metrics.
        '''
        history = {'loss': [], 'val_loss': []}
        for metric in self.metrics:
            history[metric.__class__.__name__] = []
            history[f'val_{metric.__class__.__name__}'] = []

        for epoch in range(epochs):
            self.decoder.train()
            epoch_loss = 0.0
            num_batches = 0

            for metric in self.metrics:
                metric.reset()

            iterator = tqdm(self.dataset, desc=f"Epoch {epoch+1}/{epochs}") if verbose else self.dataset

            for batch_idx, (images, labels) in enumerate(iterator):
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                # Pass labels for correct gradient computation with ground truth class
                feature_maps, target_images = self._process_batch(images, labels=labels, use_augmentation=True)

                self.optimizer.zero_grad()
                outputs = self.decoder(feature_maps)
                loss = self.criterion(outputs, target_images)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                with torch.no_grad():
                    for metric in self.metrics:
                        metric.update(target_images, outputs)

                if verbose and batch_idx % 10 == 0:
                    iterator.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / max(num_batches, 1)
            history['loss'].append(avg_loss)

            for metric in self.metrics:
                history[metric.__class__.__name__].append(metric.compute())

            if self.valid_dataset is not None:
                val_loss, val_metrics = self._validate()
                history['val_loss'].append(val_loss)
                for i, metric in enumerate(self.metrics):
                    history[f'val_{metric.__class__.__name__}'].append(val_metrics[i])

            # Step the learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = None

            if verbose:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}"
                for metric in self.metrics:
                    msg += f" - {metric.__class__.__name__}: {history[metric.__class__.__name__][-1]:.4f}"
                if self.valid_dataset is not None:
                    msg += f" - Val Loss: {history['val_loss'][-1]:.4f}"
                if current_lr is not None:
                    msg += f" - LR: {current_lr:.2e}"
                print(msg)

        return history

    def _validate(self):
        """Run validation."""
        self.decoder.eval()
        val_loss = 0.0
        num_batches = 0

        val_metrics = [metric.__class__() for metric in self.metrics]

        with torch.no_grad():
            for images, labels in self.valid_dataset:
                images = images.to(self.device).float()
                labels = labels.to(self.device)

                # Pass labels for correct gradient computation
                feature_maps, target_images = self._process_batch(images, labels=labels, use_augmentation=False)
                outputs = self.decoder(feature_maps)
                loss = self.criterion(outputs, target_images)

                val_loss += loss.item()
                num_batches += 1

                for metric in val_metrics:
                    metric.update(target_images, outputs)

        avg_val_loss = val_loss / max(num_batches, 1)
        val_metric_values = [metric.compute() for metric in val_metrics]

        return avg_val_loss, val_metric_values

    def train(self, *args, **kwargs):
        '''Alias for fit method.'''
        return self.fit(*args, **kwargs)

    def __call__(self, image, class_idx=None, percentile=95):
        '''
        Generates visual explanations for a given image.

        Parameters:
            image: torch.Tensor
                The input image or batch of images. Shape: (C, H, W) or (N, C, H, W).
                Values should be in [0, 255] range.
            class_idx: int or None
                The class index for which the explanation is generated. Default is None.
            percentile: int
                The percentile value P for gradient masking (Eq. 1).
                Default is 95 as recommended in the paper for inference.

        Returns:
            tuple: A tuple containing:
                - confidence_maps: Self-Confidence Map (Ĉ) indicating important regions
                - recovered_images: Reconstructed images (Ŷr) from the decoder
        '''
        self.decoder.eval()

        nobatch = len(image.shape) == 3

        if nobatch:
            image = image.unsqueeze(0)
            if class_idx is not None:
                class_idx = torch.tensor([class_idx], device=self.device)

        image = image.to(self.device)

        image = F.interpolate(image, size=self.IMG_SIZE, mode='bilinear', align_corners=False)

        gradient_masked_feature_maps = []

        for i in range(image.shape[0]):
            img = image[i:i+1]
            cidx = class_idx[i].item() if class_idx is not None else None

            feature_map, gradient_map = self._gradient_map(img, cidx)

            if feature_map.dim() == 3:
                feature_map = feature_map.unsqueeze(0)
                gradient_map = gradient_map.unsqueeze(0)

            flat_gradient_map = gradient_map.reshape(gradient_map.shape[0], -1)

            for j, gmap in enumerate(flat_gradient_map):
                Qmask = gmap[gmap > 0]
                if len(Qmask) > 0:
                    Q = self._nanpercentile(Qmask, (percentile, percentile))
                else:
                    Q = 0

                gmask_map = (gradient_map[j] >= Q).float()
                gradient_masked_feature_maps.append(feature_map[j] * gmask_map)

        gradient_masked_feature_maps = torch.stack(gradient_masked_feature_maps)

        with torch.no_grad():
            decoded_representations = self.decoder(gradient_masked_feature_maps)

        confidence_maps = self._stretching_sine_function(decoded_representations[:, 0])
        recovered_images = ((decoded_representations[:, 1:] + 1) * 127.5).clamp(0, 255).byte()

        recovered_images = recovered_images.permute(0, 2, 3, 1)

        if nobatch:
            confidence_maps = confidence_maps[0]
            recovered_images = recovered_images[0]

        return confidence_maps.cpu(), recovered_images.cpu()
