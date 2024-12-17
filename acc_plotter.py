import matplotlib.pyplot as plt

from training_monitor import TrainingMonitor
from validation_monitor import ValidationMonitor
from config import Config

class AccPlotter:
    def __init__(self, training_monitor: TrainingMonitor, val_monitor: ValidationMonitor) -> None:
        self.training_monitor = training_monitor
        self.val_monitor = val_monitor

    def plot_acc(self) -> None:
        train_acc: list[float] = self.training_monitor.get_train_acc()
        val_acc: list[float] = self.val_monitor.get_val_acc()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, Config.EPOCHS + 1), train_acc, label='Train Accuracy', color='blue', linestyle='-', marker='o')
        plt.plot(range(1, Config.EPOCHS + 1), val_acc, label='Validation Accuracy', color='red', linestyle='--', marker='x')

        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.legend()

        plt.savefig(Config.ACC_PLOT_PATH)

        plt.show()