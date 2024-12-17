import torch
import torchvision
from torchvision.models import VGG
import torchsummary

from config import Config

class VggTrainer:
    
    def build_vgg_model(self) -> None:
        self.model: VGG = torchvision.models.vgg19_bn(num_classes=Config.CLASS_NUM)
        torch.save(self.model, Config.MODEL_PATH)
    
    def show_vgg_structure(self) -> None:
        torchsummary.summary(self.model.cuda(), Config.INPUT_SHAPE)