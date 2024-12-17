from abc import ABC, abstractmethod

from PyQt5.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSizePolicy, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib import image

from augmented_images import ImageAugmentationShower
from training import VggTrainer
from config import Config


class Gui(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("VGG19 CIFAR10 Classifier")
        self.setGeometry(100, 100, 550, 300)

        main_layout = QHBoxLayout()

        display_column = DisplayColumn(self)
        button_column = ButtonColumn(self, display_column)

        main_layout.addWidget(button_column.create_column(), 1)
        main_layout.addWidget(display_column.create_column(), 2)
        self.setLayout(main_layout)
        return None


class BaseColumn(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_column(self) -> QGroupBox:
        pass


class ButtonColumn(BaseColumn):
    def __init__(self, parent_widget, display_column) -> None:
        self._parent_widget = parent_widget
        self.display_column = display_column
        self.vgg_trainer = VggTrainer()
        self._inference_image_path: str = None
        
    def create_column(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout()

        load_image_button = QPushButton("Load Image")
        load_image_button.clicked.connect(lambda: self._handle_load_image())
        layout.addWidget(load_image_button)

        show_augmentation_images_button = QPushButton("Show Augmentation Images")
        show_augmentation_images_button.clicked.connect(lambda: self._handle_show_augmentation_images())
        layout.addWidget(show_augmentation_images_button)

        show_model_structure_button = QPushButton("Show Model Structure")
        show_model_structure_button.clicked.connect(lambda: self._handle_show_model_structure())
        layout.addWidget(show_model_structure_button)

        show_accuracy_loss_button = QPushButton("Show Accuracy and Loss")
        show_accuracy_loss_button.clicked.connect(lambda: self._handle_show_accuracy_loss())
        layout.addWidget(show_accuracy_loss_button)

        inference_button = QPushButton("Inference")
        layout.addWidget(inference_button)

        layout.addStretch()

        group.setLayout(layout)
        
        group.setStyleSheet("QGroupBox { border: none; }")

        return group
    
    def _handle_show_augmentation_images(self) -> None:
        images, image_names = ImageAugmentationShower.augment_images(Config.AUGMENTATION_IMAGE_FOLDER_PATH)
        ImageAugmentationShower.show_images(images, image_names)
    
    def _handle_show_model_structure(self) -> None:
        self.vgg_trainer.show_vgg_structure()
    
    def _handle_show_accuracy_loss(self) -> None:
        acc_img = image.imread(Config.ACC_PLOT_PATH)
        val_img = image.imread(Config.LOSS_PLOT_PATH)

        _, axes = plt.subplots(2, 1, figsize=(6, 9))
        axes[0].imshow(acc_img)
        axes[0].axis('off')
        axes[0].set_title('Accuracy')

        axes[1].imshow(val_img)
        axes[1].axis('off')
        axes[1].set_title('Loss')

        plt.show()
        
    def _handle_load_image(self) -> None:
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self._parent_widget, "Open Image", "", "PNG Files (*.png);;All Files (*)", options=options)
        
        if file:
            self._inference_image_path = file
            print(f"Selected Image Path: {self._inference_image_path}")
            self._show_inference_image(self._inference_image_path)
    
    def _show_inference_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(128, 128, Qt.KeepAspectRatio)
        self.display_column.image_label.setPixmap(pixmap)
        self.display_column.image_label.setAlignment(Qt.AlignLeft)

class DisplayColumn(BaseColumn):
    def __init__(self, parent_widget) -> None:
        self._parent_widget = parent_widget

    def create_column(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout()

        # Inference picture show space
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: none;")

        layout.addWidget(self.image_label)

        # predict accuracy show space
        predict_label = QLabel("Predicted:")
        predict_label.setAlignment(Qt.AlignLeft)

        layout.addWidget(predict_label)
        layout.addStretch()

        group.setLayout(layout)
        group.setStyleSheet("QGroupBox { border: none; }")

        return group