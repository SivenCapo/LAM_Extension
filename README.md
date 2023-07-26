# LAM Visualization Script Extension Guide

This document provides a guide on how to extend the LAM visualization script to adapt to various models and generate LAM visualizations for the desired model. Before proceeding with the extension, make sure you are familiar with the basic usage of the LAM visualization script.

## 1. Add Custom Model Architecture

Add your own custom model architecture under the path
```LAM_Demo/ModelZoo/NN/```
and ensure that you assign the `args` parameters from your training configuration file to the model class.

## 2. Add Model Names and Weight File Names

In the 
```LAM_Demo/ModelZoo/__init__.py```
file, add the model names and their corresponding weight file names for which you want to generate LAM visualizations. Define a dictionary here to map model names to their respective weight file names.

## 3. Download the Models

Due to the LFS capacity limitation, we are unable to provide weight files for all models in the project repository.But you can obtain all the models from the website :

```https://drive.google.com/drive/folders/1nyL3gGAWeFJnhrlVIU_Ab6Y89-0qhNa0 and store them in the directory LAM_Demo/ModelZoo/models.```
and store them in the directory 
```LAM_Demo/ModelZoo/models```

## 4. Include Model Weight Files

Place the model weight files (in .pth or .pt format) that you trained in the "LAM_Demo/ModelZoo/models" directory.

## 5. Handle Mismatched Weight File Fields

When the state_dict in the weight file saved from your model training does not match the defined structure in the LAM script, you need to handle the situation accordingly. Here are some possible scenarios and their solutions:

### Scenario 1: Fields with 'module.' Prefix due to Distributed Training

Solution: You can use the following code in
```LAM_Demo/ModelZoo/__init__.py```
 to remove the 'module.' prefix and match the model's fields:

```python
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
```


### Scenario 2: Weight File Dictionary Structure Mismatch

Solution: Assuming the weight file has fields ['epoch', 'model_state_dict'], you can modify the state_dict using the following code:

```python
del state_dict['epoch']
state_dict = state_dict['model_state_dict']
```


### Scenario 3: Further Debugging

If the above methods still do not resolve the issue, you can print out the state_dict for further debugging. This will help you identify and resolve issues related to the model weight file.

## 5. Pay Attention to Input Color Format

When using the LAM script, the input image color format is (0, 1). If your model expects input in (0, 255) or any other range, normalize it to the (0, 1) range before running the LAM script to ensure smooth operation.

After completing the above steps, your model will successfully adapt to the LAM visualization script and generate LAM visualizations for the desired model. Best of luck! For any issues, refer to the relevant documentation in the LAM visualization script or perform further debugging.
