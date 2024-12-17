class Config:
    AUGMENTATION_IMAGE_FOLDER_PATH: str = ".\\datasets\\aumentation_dataset"
    
    CLASS_NUM: int = 10
    
    MODEL_PATH: str = ".\\models\\vgg19.pth"
    
    INPUT_SHAPE: tuple[int, int, int] = (3, 32, 32)