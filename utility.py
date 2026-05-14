import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader


def display_image(img):
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)


def visualize_triplets(triplet_dataset: Dataset, index: int = 1):
    fig = plt.figure(figsize=(15, 6))
    anchor, positive, negative = triplet_dataset[index]
    ax = fig.add_subplot(1, 3, 1)
    display_image(anchor)
    ax.set_xlabel("Anchor")
    ax = fig.add_subplot(1, 3, 2)
    display_image(positive)
    ax.set_xlabel("Positive")
    ax = fig.add_subplot(1, 3, 3)
    display_image(negative)
    ax.set_xlabel("Negative")
