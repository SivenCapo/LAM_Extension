# LAM Visualization Script Extension Guide
LAM (Latent Attention Maps) is an interpretable technique used to understand the decision-making process of deep neural network models. It visualizes the attention distribution within the neural network, helping us comprehend the important regions and features that the model focuses on while processing input data. The generation of LAMs is based on input samples and target classes, providing insights into the decision process of the model under various input conditions.
![image](https://github.com/SivenCapo/LAM_Extension/assets/140587950/ae02f5a5-297e-47d9-8a24-b5381b7f5dbe)

LAMs are generated through backpropagation of gradients. During the inference stage, we feed the input sample into the neural network and obtain the predicted output. Then, by calculating the gradients of the loss function with respect to the input sample, we obtain the gradient values for each pixel. These gradient values reflect the model's attention level at each position of the input sample.

This document provides a guide on how to extend the LAM visualization script to adapt to various models and generate LAM visualizations for the desired model. （I have resolved the issues that were preventing the original version from running successfully :P
Before proceeding with the extension, make sure you are familiar with the basic usage of the LAM visualization script.

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

[https://drive.google.com/drive/folders/1nyL3gGAWeFJnhrlVIU_Ab6Y89-0qhNa0 and store them in the directory LAM_Demo/ModelZoo/models]

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

# Examples:
## LAM for SCET：
![8b9dbc315ae591c3124f126752b520b](https://github.com/SivenCapo/LAM_Extension/assets/140587950/c8704b8b-c6da-4ea0-82f3-d3da42ca533c)

## LAM for ELAN：
![7005dba81bb6536c2a2ee3567b0f8dc](https://github.com/SivenCapo/LAM_Extension/assets/140587950/d2a1aba6-8a6a-426b-b862-e25f3839f5ad)

## LAM for SwinIR：
![c5ea1e672a0a2dceb2626eb7b977d6c](https://github.com/SivenCapo/LAM_Extension/assets/140587950/0a26ded3-ef36-4068-a640-03f59eb475a1)
