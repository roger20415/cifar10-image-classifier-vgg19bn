import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from config import Config

class VggInferencer:
    def inference(self, image_path: str) -> str:
        
        model = torch.load(Config.MODEL_PATH)
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        image = Image.open(image_path).convert("RGB")
        
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            predict = model(image)
        probs: np.ndarray = torch.nn.functional.softmax(predict[0], dim=0).detach().cpu().numpy()
        _, predicted_idx = torch.max(predict, 1)
        predicted_label: str = Config.CLASSES[predicted_idx.item()]
        
        self._save_probability_bar(probs)
        
        return predicted_label
    
    def _save_probability_bar(self, probs: np.ndarray) -> None:
        plt.bar(Config.CLASSES, probs)
        plt.title('Probability of each class')
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.savefig(Config.PROBABILITY_BAR_PATH)
        plt.show()
        
        