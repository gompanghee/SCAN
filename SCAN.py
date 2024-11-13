"""
COPYRIGHT NOTICE

© 2024 Author #15213 in CVPR 2025. All rights reserved.

This software and its associated documentation files (the "Software") are owned by Author #15213 in CVPR 2025. 
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
"""


import tensorflow as tf # tested on v2.10.0
import matplotlib.pyplot as plt # tested on v3.6.1
import tensorflow_datasets as tfds # tested on v4.9.2
import numpy as np # tested on v1.24.4


class ResidualModule(tf.keras.layers.Layer):
    def __init__(self, filters=256, kernel_size=3, strides=1, padding='VALID', activation="ReLU", dilation_rate=1, *args, **kwards):
        super().__init__(*args, **kwards)
        self.filters=filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.dilation_rate = dilation_rate
        
        self.conv0=tf.keras.layers.Conv2D(filters=filters, 
                          kernel_size=1, 
                          strides=strides,
                          padding=padding,
                          activation='linear',
                          dilation_rate=1,
                          name="Residual/Conv0")
        self.bn0=tf.keras.layers.BatchNormalization(name="Residual/BN0")
        self.act0=tf.keras.layers.Activation(activation,
                             name="Residual/Act0")
        
        self.conv1=tf.keras.layers.Conv2D(filters=filters//2, 
                          kernel_size=1, 
                          strides=1,
                          padding="SAME",
                          activation='linear',
                          dilation_rate=1,
                          name="Residual/Conv1")
        self.bn1=tf.keras.layers.BatchNormalization(name="Residual/BN1")
        self.act1=tf.keras.layers.Activation(activation,
                             name="Residual/Act1")
        
        self.conv2=tf.keras.layers.Conv2D(filters=filters/2, 
                          kernel_size=kernel_size, 
                          strides=strides,
                          padding=padding,
                          activation='linear',
                          dilation_rate=dilation_rate,
                          name="Residual/Conv2")
        self.bn2=tf.keras.layers.BatchNormalization(name="Residual/BN2")
        self.act2=tf.keras.layers.Activation(activation,
                             name="Residual/Act2")
        
        self.conv3=tf.keras.layers.Conv2D(filters=filters, 
                          kernel_size=1, 
                          strides=1,
                          padding="SAME",
                          activation='linear',
                          dilation_rate=1,
                          name="Residual/Conv3")
        self.bn3=tf.keras.layers.BatchNormalization(name="Residual/BN3")
        self.act3=tf.keras.layers.Activation(activation,
                             name="Residual/Act3")
        
        self.add=tf.keras.layers.Add(name="Residual/Add")
        
    def build(self, input_shape):
        super().build(self)
        self.batch, self.height, self.width, self.channel=input_shape

    def call(self, inputs):
        height=self.height
        width=self.width
        channel=self.channel
        kernel_size=self.kernel_size
        strides=self.strides
        padding=self.padding.upper()
        dilation_rate=self.dilation_rate
        
        if channel == self.filters:
            x=inputs
        else:
            x=self.conv0(self.act0(self.bn0(inputs)))
        x_=x

        x=self.conv1(self.act1(self.bn1(x)))
        x=self.conv2(self.act2(self.bn2(x)))
        x=self.conv3(self.act3(self.bn3(x)))
        
        x=self.add([x_,x])
        
        return x
        
    
    def get_config(self):
        config = super().get_config().copy()
        
        config.update({"conv0":self.conv0,
                       "bn0":self.bn0,
                       "act0":self.act0,
                       "conv1":self.conv1,
                       "bn1":self.bn1,
                       "act1":self.act1,
                       "conv2":self.conv2,
                       "bn2":self.bn2,
                       "act2":self.act2,
                       "conv3":self.conv3,
                       "bn3":self.bn3,
                       "act3":self.act3,
                       "add":self.add})
        
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim*4, activation=tf.nn.gelu),
            tf.keras.layers.Dense(emb_dim, activation='linear')
        ])
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.ln1(x + self.attn(x, x))
        x = self.ln2(x + self.ffn(x))
        return x

class ConfMSELoss(tf.keras.losses.Loss):
    def __init__(self, alpha=4.0):
        super().__init__()
        self.alpha = alpha
        self.conf_lowest_point = 1/(1+self.alpha)

    def _stretching_sine_function(self, x, alpha=1.5, cycle_factor=8.0):
        return ((x/tf.maximum(tf.abs(x),1e-13))*tf.sin((2*np.pi*tf.abs(x))/tf.maximum(cycle_factor+alpha*tf.abs(x),1e-13))+1.)/2. 

    def call(self, y_true, y_pred):
        conf = self._stretching_sine_function(y_pred[...,0:1])
        pred = y_pred[...,1:]

        sq_err = tf.maximum(tf.abs(y_true - pred),1e-13)**2

        omega = tf.maximum(tf.reduce_mean(conf, axis=[-3,-2,-1], keepdims=True) ,1e-13)
        omega = tf.abs(omega - self.conf_lowest_point)**2/tf.maximum(omega*(1-omega), 1e-13)

        loss_conf = (1 + omega) * (tf.reduce_mean(sq_err, axis=[-3, -2, -1]) + 0.1) - 0.1
        loss_recovery = tf.reduce_mean((conf / self.alpha) * sq_err 
                                       + (1-conf) * sq_err
                                       , axis=[-3, -2, -1])

        return tf.reduce_mean(loss_conf + loss_recovery)

class ConfMAE_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='Confident_MAE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')
        self.cnt = self.add_weight(name='cnt', initializer='zeros')


    def _stretching_sine_function(self, x, alpha=1.5, cycle_factor=8.0):
        return ((x/tf.maximum(tf.abs(x),1e-13))*tf.sin((2*np.pi*tf.abs(x))/tf.maximum(cycle_factor+alpha*tf.abs(x),1e-13))+1.)/2. 


    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(tf.shape(y_true)[0]):
            conf = self._stretching_sine_function(y_pred[...,0:1])
            pred = y_pred[...,1:]

            value = conf * tf.abs(y_true - pred)
            value = tf.reduce_sum(value)/tf.reduce_sum(conf)

            self.value.assign_add(value)
            self.cnt.assign_add(1.)

    def result(self):
        return self.value/tf.maximum(self.cnt,1e-17)

    def reset_state(self):
        self.value.assign(0)
        self.cnt.assign(0)

class NoConfMAE_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='Not_Confident_MAE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = self.add_weight(name='value', initializer='zeros')
        self.cnt = self.add_weight(name='cnt', initializer='zeros')

    def _stretching_sine_function(self, x, alpha=1.5, cycle_factor=8.0):
        return ((x/tf.maximum(tf.abs(x),1e-13))*tf.sin((2*np.pi*tf.abs(x))/tf.maximum(cycle_factor+alpha*tf.abs(x),1e-13))+1.)/2. 


    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(tf.shape(y_true)[0]):
            conf = self._stretching_sine_function(y_pred[...,0:1])
            pred = y_pred[...,1:]

            value = (1-conf) * tf.abs(y_true - pred)
            value = tf.reduce_sum(value)/tf.reduce_sum(1-conf)

            self.value.assign_add(value)
            self.cnt.assign_add(1.)

    def result(self):
        return self.value/tf.maximum(self.cnt,1e-17)

    def reset_state(self):
        self.value.assign(0)
        self.cnt.assign(0)

class SCAN:
    '''
    SCAN Library

        The SCAN library is designed to extract visual explanations from deep learning models. By specifying the model and layer to be analyzed, it learns to analyze visual explanations using the feature map extracted from the specified layer. Once trained, it can output visual explanations for all input images.

        Steps to use this library:

        1. Specify the target model and layer. It works with most models and layers.
        2. Specify the training dataset and validation dataset (optional). The dataset should be the same as the one used to train the target model.
        3. Create the decoder model. It is categorized into convolutional networks and transformer networks. This distinction is based on the different forms of intermediate feature maps. For undefined neural network structures, select based on the shape of the feature map.
        4. Compile the decoder. Specify the Alpha value used in the loss function, and define the optimizer, learning rate, and metrics for training the decoder model.
        5. Train the decoder. Use the fit or train function to start training the decoder. It begins to learn how to analyze the feature map of the target model.
        6. SCAN inference. By inputting an image, the visual explanation for that image is output.
    '''
    
    def __init__(self, target_model, target_layer, image_size=(224,224), use_gradient_mask=True, decoder_model=None):
        '''
        Parameters:
            target_model: <tf.keras.models.Model> or a Tensorflow model. 
                The model object to be analyzed.
            target_layer: <int> or <str>. 
                If specified as an int, it designates the layer by its index. If specified as a str, it designates the layer by its name.
            image_size: tuple or None.
                If set as a tuple, it works as its image size. If set as a None, it extracts the size from target_model.
            use_gradient_mask: <True> or <False>. 
                Specifies whether to use gradient masking during training.
            decoder_model: <tf.keras.models.Model> or None. 
                If an existing decoder model is available, it can be used by inputting it as this parameter.

        '''
        
        self.model = target_model
        self.decoder = decoder_model
        self.target_layer = target_layer
        self.use_gradient_mask = use_gradient_mask
        
        self.valid_dataset=None
        
        self.preprocess = lambda x:x
        
        if image_size is None:
            self.IMG_SIZE = self.model.input.shape[1:3]
        else:
            self.IMG_SIZE = image_size
        
        
        if type(target_layer) is str:
            _target_layer=self.model.get_layer(target_layer).output
        else:
            _target_layer=self.model.layers[target_layer].output
            
            
        if type(_target_layer) is tuple:
            _target_layer=_target_layer[0]
                
                
        self.grad_model = tf.keras.Model(inputs=self.model.inputs, outputs=[_target_layer, self.model.output])
        
        
    def _stretching_sine_function(self, x, alpha=1.5, cycle_factor=8.0):
        return ((x/tf.maximum(tf.abs(x),1e-13))*tf.sin((2*np.pi*tf.abs(x))/tf.maximum(cycle_factor+alpha*tf.abs(x),1e-13))+1.)/2. 
    
            
    def _nanpercentile(self, x, percentage_range=(0,95), axis=-1):
        q = percentage_range[0] + np.random.rand() * (percentage_range[1] - percentage_range[0])
        return np.nan_to_num(np.nanpercentile(x, q, axis=axis, method='closest_observation').astype(np.float32))
        
        
    def _gradient_map(self, X, class_idx=None):
        nobatch=len(tf.shape(X))==3
        
        if nobatch:
            X=X[None]
            if class_idx is not None:
                class_idx=tf.convert_to_tensor(class_idx)[None]
        
        if self.use_gradient_mask:
            with tf.GradientTape(persistent=True) as tape:
                grad_out, pred = self.grad_model(self.preprocess(X))

                if class_idx is None:
                    class_idx=tf.argmax(pred,-1)

                pred_mask=tf.one_hot(class_idx,1000)>0.5
                pred_c=tf.boolean_mask(pred, pred_mask)

            gradmap=tape.gradient(pred_c, grad_out)

            gradmap=tf.maximum(gradmap, 0)

            if nobatch: 
                grad_out=grad_out[0]
                gradmap=gradmap[0]

            return grad_out, gradmap
        else:
            grad_out, _ = self.grad_model(self.preprocess(X))
            
            return grad_out, tf.ones_like(grad_out, dtype=tf.float32)
        
    
    def _process_data_for_decoder(self, image, label):
        image = tf.image.resize(image, self.IMG_SIZE)
        label = tf.one_hot(label,1000)
        
        feature_map, gradient_map = self._gradient_map(image)
        
        if self.use_augmentation == False:
            
            gmask_map=tf.cast(gradient_map>0, tf.float32)
            
        else:
            
            Qmask=tf.boolean_mask(gradient_map, gradient_map>0)
            Q=tf.numpy_function(self._nanpercentile, inp=[Qmask, self.use_augmentation, -1], Tout=tf.float32)

            gmask_map=tf.cast(gradient_map>=Q, tf.float32)
    
        gradient_masked_feature_maps = feature_map * gmask_map
        
        return gradient_masked_feature_maps, tf.cast(image,tf.float32)/127.5-1.
    
    
    
    def _process_valid_data_for_decoder(self, image, label):
        image = tf.image.resize(image, self.IMG_SIZE)
        label = tf.one_hot(label,1000)
        
        feature_map, gradient_map = self._gradient_map(image)
        
        gmask_map=tf.cast(gradient_map>0, tf.float32)
    
        gradient_masked_feature_maps = feature_map * gmask_map
        
        return gradient_masked_feature_maps, tf.cast(image,tf.float32)/127.5-1.
    
    
        
    def load_decoder(self, filepath, compile=False):
        '''
        Load a trained decoder model file.

        This function loads a pre-trained decoder model from the specified file path, 
        preparing it for use.

        Parameters:
            filepath: str
                The path to the decoder model file. Providing the file path allows the model to be loaded and ready for use.
            compile: bool
                Whether to compile the model. It saves elements like the optimizer and loss used during training. 
                However, due to the frequent use of custom objects, there is a high possibility that it might not work correctly.
        '''
        self.decoder = tf.keras.models.load_model(filepath, compile=compile)
        
        
    def save_decoder(self, filepath):
        '''
        Saves a trained decoder model to a file.

        Parameters:
            filepath: str
                The path where the decoder model will be saved.
        '''
        tf.keras.models.save_model(self.decoder, filepath)
        
        
    def set_preprocess(self, func):
        '''
        Sets the preprocessing function used by the target model.

        Parameters:
            func: function
                The function used for preprocessing. If the target model uses preprocessing and it is not specified, the analysis may not be accurate.

        '''
        
        self.preprocess = func
        
        return self
        
        
    def set_dataset(self, X, Y=None, batch_size=32, is_tensorflow_dataset=True, use_augmentation=(0, 95)):
        '''
            Sets the dataset for training the decoder model.

            Parameters:
                X: tf.data.Dataset or iterable 
                    The input data. If Y is None, X can be a TensorFlow dataset or an iterable of (input, label) pairs.
                Y: iterable, optional
                    The labels for the input data. If provided, X should be an iterable of inputs.
                batch_size: int, optional
                    The size of the batches in which the data will be processed. Default is 32.
                is_tensorflow_dataset: bool, optional
                    Whether the input data X is a TensorFlow dataset. Default is True.
                use_augmentation: tuple or bool, optional
                    Whether to enable augmentation. Set the Percentile Augmentation range with a tuple. Set to False if you don't want to enable it. Default is (0, 95).

            Returns:
                self: The instance of the class.
        '''
        self.use_augmentation=use_augmentation
        
        if Y is None:
            
            if is_tensorflow_dataset:
                
                self.dataset = X.map(self._process_data_for_decoder).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
            else:
                
                def _gen_data():
                    for x,y in X:
                        yield x,y
                
                dataset = tf.data.Dataset.from_generator(_gen_data)
                
                self.dataset = dataset.map(self._process_data_for_decoder).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        else:
            
            def _gen_data():
                for x,y in zip(X,Y):
                    yield x,y

            dataset = tf.data.Dataset.from_generator(_gen_data)
            self.dataset = dataset.map(self._process_data_for_decoder).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        return self
    
    
    def set_validation_dataset(self, X, Y=None, batch_size=32, is_tensorflow_dataset=True):
        '''
        Sets the dataset for evaluating the decoder model.

        Parameters:
            X: tf.data.Dataset or iterable 
                The input data. If Y is None, X can be a TensorFlow dataset or an iterable of (input, label) pairs.
            Y: iterable, optional
                The labels for the input data. If provided, X should be an iterable of inputs.
            batch_size: int, optional
                The size of the batches in which the data will be processed. Default is 32.
            is_tensorflow_dataset: bool, optional
                Whether the input data X is a TensorFlow dataset. Default is True.
            use_augmentation: tuple or bool, optional
                Whether to enable augmentation. Set the Percentile Augmentation range with a tuple. Set to False if you don't want to enable it. Default is (0, 95).

        Returns:
            self: The instance of the class.
        '''
        if Y is None:
            
            if is_tensorflow_dataset:
                
                self.valid_dataset = X.map(self._process_valid_data_for_decoder).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
            else:
                
                def _gen_data():
                    for x,y in X:
                        yield x,y
                
                dataset = tf.data.Dataset.from_generator(_gen_data)
                
                self.valid_dataset = dataset.shuffle(1024).map(self._process_valid_data_for_decoder).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        else:
            
            def _gen_data():
                for x,y in zip(X,Y):
                    yield x,y

            dataset = tf.data.Dataset.from_generator(_gen_data)
            self.valid_dataset = dataset.shuffle(1024).map(self._process_valid_data_for_decoder).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
        return self
    

    def _get_convolutional_model(self, channels=[512,384,256,192,128,64]):
        input_shape = self.grad_model.output[0].shape[1:]
            
        depth = int(np.log(self.IMG_SIZE[0]/input_shape[0])/np.log(2))-1

        inputs = tf.keras.layers.Input(input_shape)
        
        x = ResidualModule(channels[4-depth],1,1,activation="ReLU",padding="SAME")(inputs)

        for i in range(depth):
            x = ResidualModule(channels[4 - depth + i],3,1,activation="ReLU",padding="SAME")(x)
            x = ResidualModule(channels[4 - depth + i],3,1,activation="ReLU",padding="SAME")(x)
            x = tf.keras.layers.Conv2DTranspose(channels[i+1],3,2,activation="ReLU",padding="SAME")(x)

        x = ResidualModule(channels[4],3,1,activation="ReLU",padding="SAME")(x)
        x = ResidualModule(channels[4],3,1,activation="ReLU",padding="SAME")(x)
        x = tf.keras.layers.Conv2DTranspose(channels[5],3,2,activation="ReLU",padding="SAME")(x)
          
        x = ResidualModule(channels[5],3,1,activation="ReLU",padding="SAME")(x)
        x = ResidualModule(channels[5],3,1,activation="ReLU",padding="SAME")(x)
          
        x = ResidualModule(channels[5],3,1,activation="ReLU",padding="SAME")(x)
             
        out = ResidualModule(4,1,1,activation="linear",padding="SAME")(x)

        return tf.keras.Model(inputs=inputs, outputs=out)
    
    
    def _get_transformer_model(self):
        input_shape = self.grad_model.output[0].shape[1:]
        
        grid_size = int(shape[-2]**0.5)
        repeat_cnt = int(np.log(self.IMG_SIZE[0]/grid_size)/np.log(2))

        inputs = tf.keras.layers.Input(shape)

        x = TransformerBlock(emb_dim=input_shape[-1], num_heads=12)(inputs)
        x = TransformerBlock(emb_dim=input_shape[-1], num_heads=12)(x)
        x = TransformerBlock(emb_dim=input_shape[-1], num_heads=12)(x)
        x = TransformerBlock(emb_dim=input_shape[-1], num_heads=12)(x)

        if grid_size**2 != shape[-2]:
            x = tf.keras.layers.Lambda(lambda x: x[:,1:])(x) # Only for class token (ViT's)
            
        x = tf.keras.layers.Reshape((grid_size,grid_size,shape[-1]))(x)

        for i in range(repeat_cnt):
            x = tf.keras.layers.Conv2DTranspose(32*2**(repeat_cnt-1-i),3,2,activation="ReLU",padding="SAME")(x)
            x = ResidualModule(32*2**(repeat_cnt-1-i),3,1,activation="ReLU",padding="SAME")(x)
            x = ResidualModule(32*2**(repeat_cnt-1-i),3,1,activation="ReLU",padding="SAME")(x)

        out = ResidualModule(4,1,1,activation="linear",padding="SAME")(x)

        return tf.keras.Model(inputs=inputs, outputs=out)
        
        
    def generate_decoder(self, is_Transformer=False):
        '''
        Generates the decoder model based on the specified type.

        Parameters:
            is_Transformer: bool, optional
                If True, generates a transformer-based decoder model. If False, generates a convolutional-based decoder model. Default is False.

        Returns:
            self: The instance of the class with the generated decoder model.
        '''
        if is_Transformer:
            self.decoder = self._get_transformer_model()
        else:
            self.decoder = self._get_convolutional_model()
        
        return self
        
        
    def compile(self, loss_alpha=4.0, optimizer=tf.keras.optimizers.Adam, learning_rate=1e-3, metrics=[ConfMAE_Metric, NoConfMAE_Metric]):
        '''
        Compiles the decoder model with the specified parameters.

        Parameters:
            loss_alpha: float, optional
                The alpha value used in the loss function. Default is 4.0.
            optimizer: tf.keras.optimizers.Optimizer, optional
                The optimizer to use for training. Default is Adam.
            learning_rate: float, optional
                The learning rate for the optimizer. Default is 1e-3.
            metrics: list, optional
                A list of metric classes to evaluate during training. Default includes ConfMAE_Metric and NoConfMAE_Metric.

        Returns:
            self: The instance of the class with the compiled decoder model.
        '''
        self.decoder.compile(optimizer = optimizer(learning_rate=learning_rate), loss = ConfMSELoss(loss_alpha), metrics = [metric() for metric in metrics])
        
        return self
        
        
    def fit(self, epochs=2, **kwargs):
        '''
        Trains the decoder model on the dataset.

        Parameters:
            epochs: int, optional
                The number of epochs to train the model. Default is 2.
            **kwargs: Additional keyword arguments passed to the `fit` method of the model.

        Returns:
            self: The instance of the class after training the decoder model.
        '''
        self.decoder.fit(self.dataset, validation_data=self.valid_dataset, epochs=epochs, **kwargs)
        
        
    def train(self, *args, **kwargs):
        '''
        Trains the decoder model on the dataset.

        Parameters:
            epochs: int, optional
                The number of epochs to train the model. Default is 2.
            **kwargs: Additional keyword arguments passed to the `train` method of the model.

        Returns:
            self: The instance of the class after training the decoder model.
        '''
        self.fit(*args, **kwargs)
        
        
    def __call__(self, image, class_idx=None, percentile=0):
        '''
        Generates visual explanations for a given image.

        Parameters:
            image: tf.Tensor or Iterables 
                The input image or batch of images.
            class_idx: int or None, optional
                The class index for which the explanation is generated. Default is None.
            percentile: int, optional
                The percentile value used for gradient masking. Default is 0.

        Returns:
            tuple: A tuple containing confidence maps and reconstructed images.
        '''
        nobatch=len(tf.shape(image))==3
        
        if nobatch:
            image=image[None]
            
            if class_idx is not None:
                class_idx=tf.convert_to_tensor(class_idx)[None]
            
        feature_map, gradient_map = self._gradient_map(image, class_idx)
        
        flat_gradient_map = tf.reshape(gradient_map,(gradient_map.shape[0],-1))
        
        gradient_masked_feature_maps = []
        
        for i, gmap in enumerate(flat_gradient_map):
            Qmask=tf.boolean_mask(gmap, gmap>0)
            Q=tf.numpy_function(self._nanpercentile, inp=[Qmask, (percentile, percentile), -1], Tout=tf.float32)

            gmask_map=tf.cast(gradient_map[i]>=Q, tf.float32)

            gradient_masked_feature_maps.append(feature_map[i] * gmask_map)
        
        gradient_masked_feature_maps = tf.stack(gradient_masked_feature_maps)
        
        decoded_representations = self.decoder(gradient_masked_feature_maps)
        
        confidence_maps = self._stretching_sine_function(decoded_representations[...,0])
        recovered_images = tf.cast((decoded_representations[...,1:]+1)*127.5, tf.uint8)
        
        if nobatch:
            confidence_maps=confidence_maps[0]
            recovered_images=recovered_images[0]
            
        return confidence_maps, recovered_images
    
