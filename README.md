# SCAN: Visual Explanations with Self-Confidence and Representation Analytical Networks

## Overview
This repository contains the implementation of SCAN (Self-Confidence and Analytic Networks), a novel method for providing detailed visual explanations in computer vision models. SCAN leverages encoded representations and Self-Confidence Maps to highlight important regions, offering more detailed insights than existing methods like GradCAM, GradCAM++, and LayerCAM.

## Repository Contents
- SCAN.py: The main implementation file for SCAN.
- SCAN_Example_Training.ipynb: Jupyter notebook for training the SCAN model.
- SCAN_Example_Testing.ipynb: Jupyter notebook for testing the SCAN model on various examples.
- SCAN_Example_Testing_with_ViT.ipynb: Jupyter notebook for testing the SCAN model with Vision Transformer (ViT).

## Files and Their Purpose
1. SCAN.py: Contains the core SCAN implementation, including functions for generating Gradient-masked Feature Maps and Self-Confidence Maps.
2. SCAN_Example_Training.ipynb: Notebook for training the SCAN model on a specified dataset. This notebook includes data preprocessing steps, model training, and saving the trained model.
3. SCAN_Example_Testing.ipynb: Notebook for testing the SCAN model on various examples to visualize the outputs. It demonstrates how to generate and interpret the Self-Confidence Maps.
4. SCAN_Example_Testing_with_ViT.ipynb: Similar to the testing notebook but specifically designed for Vision Transformer (ViT) models.

## Usage
1. **Training the SCAN Model:**
   - Open the SCAN_Example_Training.ipynb notebook.
   - Follow the steps to preprocess the data, initialize the model, and train it.
   - Save the trained model for later use.

2. **Testing the SCAN Model:**
   - Open the SCAN_Example_Testing.ipynb notebook.
   - Load the trained model and run the test cells to generate visual explanations using SCAN.
   - Interpret the Self-Confidence Maps to understand the model's focus areas.

3. **Testing with Vision Transformer:**
   - Open the SCAN_Example_Testing_with_ViT.ipynb notebook.
   - Follow the steps to test the SCAN model specifically with Vision Transformer (ViT) models.

## Model Files
We provide around 76 trained model files that can be downloaded from the following link:

[Download SCAN Model - ResNet50v2](https://drive.google.com/file/d/1bAg1_NKDapTNLfjUH4H3Rt3KK4w84V9q/view?usp=share_link)

[Download SCAN Model - EfficientNetV2 B0](https://drive.google.com/file/d/1WK3cic2flAF1MKYvmGR6UzVd6URFsHzR/view?usp=share_link)

[Download SCAN Model - ConvNeXt Small](https://drive.google.com/file/d/1BGQo9jdxG-Pv99dcc6fsO28_Qxz12aG-/view?usp=share_link)

[Download SCAN Model - MobileNetV3 Small](https://drive.google.com/file/d/1H5_jE6dE-805Etqg8RxRYKABipn2rC_7/view?usp=share_link)

[Download SCAN Model - ViT B16](https://drive.google.com/file/d/1onoT9Q6Rjz1n8WLeD-lCFpXoEeKbyxrY/view?usp=share_link)


## Example Usage
Here is a brief example of how to use the SCAN model:

1. **Install dependencies:**
   pip install -r requirements.txt

2. **Run the training notebook:**
   - Open SCAN_Example_Training.ipynb in Jupyter Notebook or JupyterLab.
   - Execute the cells to preprocess data, train the model, and save it.

3. **Run the testing notebook:**
   - Open SCAN_Example_Testing.ipynb in Jupyter Notebook or JupyterLab.
   - Load the trained model and run the cells to visualize the Self-Confidence Maps.
  
4. **Or see the following simple example:**
  ``` python3
  from SCAN import SCAN

  target_model=tf.keras.applications.MobileNetV3Small(input_shape=(224,224,3)) # or your own here

  scanner=SCAN(target_model = target_model, target_layer = 228)\
                  .set_preprocess(tf.keras.applications.mobilenet_v3.preprocess_input)\
                  .set_dataset(train_ds).set_validation_dataset(valid_ds)\
                  .generate_decoder(is_Transformer=False)\
                  .compile(loss_alpha=4.0)

  scanner.fit(2)

  self_confidence_map, reconstructed_image = scanner(image, percentile=0)
  ```

## License
This project is licensed under the CC-BY-NC-SA license. See the LICENSE file for details.

## Contact
For any questions or issues, please open an issue on the GitHub repository.
