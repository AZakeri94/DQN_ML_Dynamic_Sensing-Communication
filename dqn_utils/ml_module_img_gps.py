# ml_module_img_gps.py
# -*- coding: utf-8 -*-
"""
Custom dataset and model for image + position → beam prediction
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import ast

# ============================
# Dataset Class
# ============================
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'unit1_rgb']
        image = Image.open(img_path).convert('RGB')
        position = ast.literal_eval(self.data.loc[idx, 'unit2_pos'])
        label = self.data.loc[idx, 'unit1_beam_32']

        if self.transform:
            image = self.transform(image)

        position = torch.tensor(position, dtype=torch.float32)

        return image, position, label

# ============================
# Model Class
# ============================
class CustomModel(nn.Module):
    def __init__(self, num_classes, pos_dim=2):
        """
        num_classes: number of beam classes
        pos_dim: dimension of position vector (default 2)
        """
        super(CustomModel, self).__init__()
        # Pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        # Remove FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # FC for concatenated features (image + position)
        self.fc = nn.Linear(512 + pos_dim, num_classes)

    def forward(self, image, position):
        #features = self.resnet(image)              # [B, 512, 1, 1]
        #features = features.view(features.size(0), -1)  # [B, 512]
        features =  image
        if features.dim() == 1:
            features = features.unsqueeze(0) 
        if position.dim() == 1: # Ensure position has batch dimension
            position = position.unsqueeze(0)
        concatenated = torch.cat((features, position), dim=1)
        output = self.fc(concatenated)
        return output

# ============================
# Optional: Main function for testing / evaluation
# ============================
def main():
    """
    Example usage of CustomDataset and CustomModel
    """
    csv_file = './scenario5_img_pos_beam_test_dqn.csv'
    
    # Preprocessing pipeline
    proc_pipe = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomDataset(csv_file, transform=proc_pipe)
    print(f"Dataset length: {len(dataset)}")
    img, pos, label = dataset[0]
    print(f"Image shape: {img.shape}, Position: {pos}, Label: {label}")
    
    # Example model
    num_classes = 33
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_dim = len(pos)
    model = CustomModel(num_classes, pos_dim=pos_dim).to(device)
    
    # Forward pass
    img = img.unsqueeze(0).to(device)   # add batch dim
    pos = pos.unsqueeze(0).to(device)
    output = model(img, pos)
    print(f"Model output shape: {output.shape}")

if __name__ == "__main__":
    main()
