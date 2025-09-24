# SCAN: Visual Explanations with Self-Confidence and Representation Analytical Networks

## Overview
This repository contains the implementation of SCAN (Self-Confidence and Analysis Networks), a novel method for providing detailed visual explanations in computer vision models. SCAN leverages encoded representations and Self-Confidence Maps to highlight important regions, offering more detailed insights than existing methods like Explainability, Rollout, GradCAM, GradCAM++, and LayerCAM.

## Repository Contents
- SCAN.py: The main implementation file for SCAN.
- SCAN_Example_Training.ipynb: Jupyter notebook for training the SCAN model.

## Files and Their Purpose
1. SCAN.py: Contains the core SCAN implementation, including functions for generating Gradient-masked Feature Maps and Self-Confidence Maps.
2. SCAN_Example_Training.ipynb: Notebook for training the SCAN model on a specified dataset. This notebook includes data preprocessing steps, model training, and saving the trained model.

## Usage
### **Training the SCAN Model:**
   - Open the SCAN_Example_Training.ipynb notebook.
   - Follow the steps to preprocess the data, initialize the model, and train it.
   - Save the trained model for later use.



## Example Usage
Here is a brief example of how to use the SCAN model:

1. **Install dependencies:**
   pip install -r requirements.txt

2. **Run the training notebook:**
   - Open SCAN_Example_Training.ipynb in Jupyter Notebook or JupyterLab.
   - Execute the cells to preprocess data, train the model, and save it.
  
3. **Try the follow simple example:**
  ``` python3
  from SCAN import SCAN

  target_model=tf.keras.applications.ResNet50V2(input_shape=(224,224,3)) # or your own here

  scanner=SCAN(target_model = target_model, target_layer = 187)\
                  .set_preprocess(tf.keras.applications.resnet_v2.preprocess_input)\
                  .set_dataset(train_ds)\
                  .set_validation_dataset(valid_ds)\
                  .generate_decoder(is_Transformer=False)\
                  .compile(loss_alpha=4.0)

  scanner.fit(5)

  self_confidence_map, reconstructed_image = scanner(image, percentile=95)
  ```

## License
This project is licensed under the CC-BY-NC-SA license. See the LICENSE file for details.

## Contact
For any questions or issues, please open an issue on the GitHub repository.
