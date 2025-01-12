# cifar10-image-classifier-vgg19bn

This project trains a VGG19 model with batch normalization (BN) using PyTorch for image classification, with a PyQt-based GUI.

## Features
![alt text](image-3.png)

1. **Load Image**  
   - Load an image for classification inference.
   ![alt text](image-4.png)

2. **Show Augmentation Images**  
   - Load CIFAR10 and show 9 augmented images with labels.
   ![alt text](image-1.png)

3. **Show Model Structure**  
   - Load VGG19 model and show model structure. 
   ![alt text](image-2.png)


4. **Show Accuracy and Loss**  
   - Show training/validating accuracy and loss.
   ![alt text](image-6.png)
   ![alt text](image-7.png)

4. **Inference**  
    - Use the model trained to run inference, show the predicted distribution and class label. 
    ![alt text](image-5.png)
    ![alt text](image.png)
    

## Requirements

- python==3.10  
- matplotlib==3.8.0
- PyQt5==5.15.11
- torchsummary==1.5.1
- opencv-python==4.8.0.74
- torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
- torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
- tqdm==4.59.0

## Installation

Using Anaconda Prompt:  

```bash
git clone https://github.com/roger20415/cifar10-image-classifier-vgg19bn.git  
cd cifar10-image-classifier-vgg19bn

conda create --name myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```
## How to Use
### Execute the program and open the GUI
```bash
# Change directory to the project folder
cd cifar10-image-classifier-vgg19bn

# Activate conda virtual environment
conda activate myenv

# Train models
python training.py

# Execute the main script to start the project
python main.py
```
### Note

- `training.py` must be executed first to generate the required models for the functionalities in GUI.
- If you start training, you can see the following log
  ![alt text](image-8.png)