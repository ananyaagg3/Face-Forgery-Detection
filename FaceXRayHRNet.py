import torch
import torch.nn as nn
from torchvision import models

class FaceXRayHRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FaceXRayHRNet, self).__init__()
        
        # Using ResNet-50 as the backbone (you can switch to other variants like ResNet-18, VGG, etc.)
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')  # Using the updated weights argument
        self.resnet.fc = nn.Identity()  # Removing the fully connected layer for feature extraction

        # Apply a fully connected layer to generate the X-ray mask
        self.fc = nn.Linear(2048, 1)  # 2048 input features, output 1 (X-ray mask)

    def forward(self, x):
        # Pass through ResNet for feature extraction
        features = self.resnet(x)  # Shape: [batch_size, 2048]
        
        # Apply fully connected layer to generate the face X-ray (greyscale mask)
        x_ray = self.fc(features)  # Shape: [batch_size, 1]
        
        # Apply a sigmoid to convert to 0-1 mask
        x_ray = torch.sigmoid(x_ray)  # Return shape: [batch_size, 1]
        
        # Optionally, reshape to [batch_size, 1, 1, 1] if needed
        return x_ray.view(x_ray.size(0), 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]
