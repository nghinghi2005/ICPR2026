from matplotlib import pyplot as plt
import numpy as np


def visualize(img: np.ndarray):
    plt.imshow(img)
    plt.axis("off")
    plt.show()