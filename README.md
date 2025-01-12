# cifar10-image-classifier-vgg19bn

This project trains a VGG19 model with batch normalization (BN) using PyTorch for image classification, with a PyQt-based GUI.

## Features
![image-3](https://github.com/user-attachments/assets/cc5d89f7-3df1-4a29-b4e9-e8ee6a446a24)


1. **Load Image**  
   - Load an image for classification inference.
   - You can choose images from `./datasets/Inference_dataset` folder. 
   ![image-4](https://github.com/user-attachments/assets/a6503d32-efa9-48dd-9be9-42bace3ee224)

2. **Show Augmentation Images**  
   - Load CIFAR10 and show 9 augmented images with labels.  
   ![image-1](https://github.com/user-attachments/assets/104f93af-3063-4e7d-882b-7329213011bf)

3. **Show Model Structure**  
   - Load VGG19 model and show model structure.  
   ![image-2](https://github.com/user-attachments/assets/0e9c018b-c598-4287-8b50-3e1917d32d26)


4. **Show Accuracy and Loss**  
   - Show training/validating accuracy and loss.  
   ![image-6](https://github.com/user-attachments/assets/b2fea84f-06b3-4905-95e7-5aee173616dc)

   ![image-7](https://github.com/user-attachments/assets/5c16b722-2b14-4777-995e-e700d3dd9e2b)

4. **Inference**  
    - Use the model trained to run inference, show the predicted distribution and class label.  
    ![image-5](https://github.com/user-attachments/assets/0eb18835-0b68-4a16-bd1c-cf58339c1e05)  
    ![image](https://github.com/user-attachments/assets/350a431d-1a41-4a7f-ac75-81ab1652c042)

    

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
  ![image-8](https://github.com/user-attachments/assets/cac5741a-2822-423f-b5a3-cba38bcb8ca9)
