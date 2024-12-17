import matplotlib.pyplot as plt

from training_monitor import TrainingMonitor
from validation_monitor import ValidationMonitor
from config import Config

class LossPlotter:
    def __init__(self, training_monitor: TrainingMonitor, val_monitor: ValidationMonitor) -> None:
        self.training_monitor = training_monitor
        self.val_monitor = val_monitor

    def plot_loss(self) -> None:
        train_loss: list[float] = self.training_monitor.get_train_loss()
        val_loss: list[float] = self.val_monitor.get_val_loss()
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, Config.EPOCHS + 1), train_loss, label='Train Loss', color='blue', linestyle='-', marker='o')
        plt.plot(range(1, Config.EPOCHS + 1), val_loss, label='Validation Loss', color='red', linestyle='--', marker='x')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.legend()

        plt.savefig(Config.LOSS_PLOT_PATH)

        plt.show()