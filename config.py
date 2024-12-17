class Config:
    AUGMENTATION_IMAGE_FOLDER_PATH: str = ".\\datasets\\aumentation_dataset"
    
    CLASSES: tuple[str] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    MODEL_PATH: str = ".\\models\\vgg19.pth"
    
    INPUT_SHAPE: tuple[int, int, int] = (3, 32, 32)
    
    
    BATCH_SIZE: int = 100
    LEARNING_RATE: float = 0.01
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 5e-4
    STEP_SIZE: int = 30
    GAMMA: float = 0.2
    EPOCHS: int = 1
    
    TRAIN_SET_PATH: str = "./datasets/train_set"
    TEST_SET_PATH: str = "./datasets/test_set"
    
    
    LOSS_PLOT_PATH: str = ".\\plots\\loss_plot.png"
    ACC_PLOT_PATH: str = ".\\plots\\acc_plot.png"
    PROBABILITY_BAR_PATH: str = ".\\plots\\probability_bar.png"