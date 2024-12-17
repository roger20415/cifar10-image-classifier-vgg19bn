import os

import torch
import torchvision
from torchvision.models import VGG
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
import torchsummary
from torch import nn
from torch import optim
from tqdm import tqdm

from config import Config
from training_monitor import TrainingMonitor
from validation_monitor import ValidationMonitor
from loss_plottor import LossPlottor

class VggTrainer:

    def __init__(self):
        
        self.training_monitor = TrainingMonitor()
        self.val_monitor = ValidationMonitor()
        self.loss_plottor = LossPlottor(self.training_monitor, self.val_monitor)
        
        if not os.path.exists(Config.MODEL_PATH):
            self._build_vgg_model()
        else:
            print(f"Model found at {Config.MODEL_PATH}. Loading model.")
    
    def show_vgg_structure(self) -> None:
        model: VGG = torch.load(Config.MODEL_PATH)
        torchsummary.summary(model.cuda(), Config.INPUT_SHAPE)
        
    def train_vgg(self) -> None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            
        transform: Compose = Compose(
            [v2.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            v2.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

        train_set = torchvision.datasets.CIFAR10(root=Config.TEST_SET_PATH, train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_SIZE,
                                                shuffle=True, num_workers=2, pin_memory=True)

        test_set = torchvision.datasets.CIFAR10(root=Config.TEST_SET_PATH, train=False,
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=Config.BATCH_SIZE,
                                                shuffle=False, num_workers=2, pin_memory=True)

        model: VGG = torch.load(Config.MODEL_PATH).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), 
                              lr=Config.LEARNING_RATE, 
                              momentum=Config.MOMENTUM, 
                              weight_decay=Config.WEIGHT_DECAY)
        
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                 step_size=Config.STEP_SIZE, 
                                                 gamma=Config.GAMMA)
        
        pbar = tqdm(range(Config.EPOCHS), desc="Epoch")
        
        self._best_val_acc: float = 0.0
        for epoch in pbar:
            self.training_monitor.epoch_reset()
            # Train
            model.train()
            for inputs, labels in tqdm(train_loader, unit="images", unit_scale=train_loader.batch_size, leave=False, desc="Train"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                self.training_monitor.add_running_loss(loss.item())
                _, preds = torch.max(outputs, 1)
                self.training_monitor.add_correct_train_num(torch.sum(preds == labels.data).item())
                self.training_monitor.add_total_train_num(inputs.size(0))
            
            self.training_monitor.append_train_loss(len(train_loader))
            self.training_monitor.append_train_acc()
            
            # Validate
            model.eval()
            self.val_monitor.epoch_reset()
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, unit="images", unit_scale=test_loader.batch_size, leave=False, desc="Test"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    self.val_monitor.add_running_val_loss(loss.item())
                    _, preds = torch.max(outputs, 1)
                    self.val_monitor.add_correct_val_num(torch.sum(preds == labels.data).item())
                    self.val_monitor.add_total_val_num(inputs.size(0))
            lr_scheduler.step()
            self.val_monitor.append_val_loss(len(test_loader))
            self.val_monitor.append_val_acc()
        
            self._save_model(model)
            
        self.loss_plottor.plot_loss()

    def _build_vgg_model(self) -> None:
        model: VGG = torchvision.models.vgg19_bn(num_classes=len(Config.CLASSES))
        torch.save(model, Config.MODEL_PATH)

    def _save_model(self, model: VGG) -> None:
        last_val_acc: float = self.val_monitor.get_last_val_acc()
        if last_val_acc > self._best_val_acc:
            model_dir = os.path.dirname(Config.MODEL_PATH)

            model_filename = f"vgg19_val_acc_{last_val_acc:.2f}.pth"
            model_path = os.path.join(model_dir, model_filename)

            torch.save(model, model_path)
            self._best_val_acc = last_val_acc
        
if __name__ == '__main__':
    trainer = VggTrainer()
    trainer.train_vgg()