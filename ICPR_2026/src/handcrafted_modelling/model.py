import cv2
import numpy as np
from typing import Optional, Union
from .base import HandCrafter
import matplotlib.pyplot as plt

class LicensePlateEnhancer(HandCrafter):

    def __init__(self, 
        scale: Optional[Union[float, int]] = 2,
        use_visualize: bool=True
    ):
        self.scale = scale
        self.use_visualize = use_visualize

    def enhance(self, name: str, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found, double check {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        upscaler = self.use_method(name, img, self.scale)
        out = upscaler()
        if self.use_visualize:
            self.visualize(out)

        return out
        
    
    def save(self, name: str, img: np.ndarray):
        # Chuyển ngược lại BGR để cv2.imwrite lưu đúng màu
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{name}.png", save_img)
        print(f"Saved {name}.png")
    
    def visualize(self, img: np.ndarray):
        plt.imshow(img)
        plt.axis("off")
        plt.show()