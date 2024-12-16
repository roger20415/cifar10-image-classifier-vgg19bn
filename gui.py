from abc import ABC, abstractmethod
from PyQt5.QtWidgets import (QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt


class Gui(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("VGG19 CIFAR10 Classifier")
        self.setGeometry(100, 100, 550, 300)

        main_layout = QHBoxLayout()

        button_column = ButtonColumn(self)
        display_column = DisplayColumn(self)

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
    def __init__(self, parent_widget) -> None:
        self._parent_widget = parent_widget

    def create_column(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout()

        load_image_button = QPushButton("Load Image")
        layout.addWidget(load_image_button)

        show_augmentation_images_button = QPushButton("Show Augmentation Images")
        layout.addWidget(show_augmentation_images_button)

        show_model_structure_button = QPushButton("Show Model Structure")
        layout.addWidget(show_model_structure_button)

        show_accuracy_loss_button = QPushButton("Show Accuracy and Loss")
        layout.addWidget(show_accuracy_loss_button)

        inference_button = QPushButton("Inference")
        layout.addWidget(inference_button)

        layout.addStretch()

        group.setLayout(layout)
        
        group.setStyleSheet("QGroupBox { border: none; }")

        return group


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