import os

from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
import matplotlib.pyplot as plt

class ImageAugmentationShower:

    @staticmethod
    def augment_images(origin_image_folder_path: str) -> tuple[list[Image.Image], list[str]]:
        image_names: list[str] = os.listdir(origin_image_folder_path)
        images: list[Image.Image] = []

        transform: Compose = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(30)
        ])

        for image_name in image_names:
            image_path: str = os.path.join(origin_image_folder_path, image_name)
            print("image_path: " + image_path)
            try:
                image = Image.open(image_path)
                image = transform(image)
                images.append(image)
                print(str(type(image)))
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")


        return images, image_names

    @staticmethod
    def show_images(images: list[Image.Image], image_names: list[str]) -> None:
        plt.figure(figsize=(12, 12))

        for i, image in enumerate(images):
            plt.subplot(4, 3, i + 1)
            plt.imshow(image)
            plt.title(image_names[i].split(".")[0])
        plt.show()