import os
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset,random_split
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torchvision import transforms, models 
import cv2  

dataset_path = r'C:\Users\Ali\Desktop\Deep Learning papers\Fruit_detection\dataset\train\train' 
transform = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.ToTensor()
    
])

class CustomDataset(Dataset):
    def __init__(self, root = dataset_path, transform=transform):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                images.append((file_path, int(class_idx)))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, label = self.images[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# Replace 'path/to/your/dataset' with the actual path to your dataset


# Define the data transformations

