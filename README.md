# SCAN: Visual Explanations with Self-Confidence and Representation Analytical Networks (PyTorch Version)

## Overview
This repository contains the **PyTorch implementation** of SCAN (Self-Confidence and Analysis Networks), a novel method for providing detailed visual explanations in computer vision models. SCAN leverages encoded representations and Self-Confidence Maps to highlight important regions, offering more detailed insights than existing methods like Explainability, Rollout, GradCAM, GradCAM++, and LayerCAM.

## Repository Contents
- `SCAN.py`: The main PyTorch implementation file for SCAN.
- `SCAN_Example_Training_PyTorch.ipynb`: Jupyter notebook for training the SCAN model with PyTorch.

## Files and Their Purpose
1. **SCAN.py**: Contains the core SCAN implementation for PyTorch, including:
   - `ResidualModule`: Residual block for the decoder
   - `TransformerBlock`: Transformer block for transformer-based decoders
   - `ConvolutionalDecoder`: Decoder for CNN-based models
   - `TransformerDecoder`: Decoder for transformer-based models (ViT, etc.)
   - `SCAN`: Main class for training and inference

2. **SCAN_Example_Training_PyTorch.ipynb**: Notebook for training the SCAN model on a specified dataset. This notebook includes:
   - Model loading and configuration
   - Dataset preparation
   - Training the decoder
   - Generating visual explanations
   - Visualization utilities

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- matplotlib >= 3.6.0
- tqdm >= 4.64.0

## Usage

### Quick Start Example

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from SCAN import SCAN

# 1. Load target model
target_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# 2. Define preprocessing function
def preprocess_input(x):
    x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

# 3. Initialize SCAN
scanner = SCAN(
    target_model=target_model,
    target_layer='layer4',  # Use layer name
    image_size=(224, 224),
    use_gradient_mask=True,
    num_classes=1000
)

# 4. Set preprocessing and datasets
scanner.set_preprocess(preprocess_input)\
       .set_dataset(train_loader, use_augmentation=(70, 100))\
       .set_validation_dataset(valid_loader)\
       .generate_decoder(is_Transformer=False)\
       .compile(loss_alpha=4.0, learning_rate=1e-3)

# 5. Train
scanner.fit(epochs=5)

# 6. Generate visual explanation
confidence_map, reconstructed_image = scanner(image, percentile=95)
```

### Key Differences from TensorFlow Version

| Feature | TensorFlow | PyTorch |
|---------|------------|---------|
| Target Layer | Layer index (int) | Layer name (str) or module reference |
| Dataset | tf.data.Dataset | torch.utils.data.DataLoader |
| Preprocessing | Function that takes tensor | Function that takes tensor |
| Image Format | (N, H, W, C) | (N, C, H, W) |
| Output | (H, W) confidence, (H, W, C) image | (H, W) confidence, (H, W, C) image |

### Target Layer Selection

For **ResNet** models:
- `'layer4'`: Last convolutional block (recommended)
- `'layer3'`: Earlier features (more detailed but noisier)
- `'layer4.2.conv3'`: Specific conv layer

For **Vision Transformer (ViT)**:
```python
scanner.generate_decoder(is_Transformer=True)
```

### Saving and Loading

```python
# Save trained decoder
scanner.save_decoder('scan_decoder.pt')

# Load decoder
scanner.load_decoder('scan_decoder.pt')
```

## API Reference

### SCAN Class

```python
SCAN(
    target_model,        # PyTorch model to analyze
    target_layer,        # Layer name (str), module reference, or index (int)
    image_size=(224, 224),  # Input image size
    use_gradient_mask=True, # Use gradient masking
    decoder_model=None,  # Pre-trained decoder (optional)
    device=None,         # 'cuda' or 'cpu' (auto-detected if None)
    num_classes=1000     # Number of classes
)
```

### Methods

- `.set_preprocess(func)`: Set preprocessing function
- `.set_dataset(dataloader, use_augmentation=(70, 100))`: Set training data
- `.set_validation_dataset(dataloader)`: Set validation data
- `.generate_decoder(is_Transformer=False, ch_per_lv=[192, 256, 384, 512])`: Create decoder model
- `.compile(loss_alpha=4.0, optimizer_class=Adam, learning_rate=1e-3)`: Compile
- `.fit(epochs=2, verbose=True)`: Train decoder
- `.__call__(image, class_idx=None, percentile=95)`: Generate explanation

## License
This project is licensed under the CC-BY-NC-SA license. See the LICENSE file for details.

## Contact
For any questions or issues, please open an issue on the GitHub repository.
