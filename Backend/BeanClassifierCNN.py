import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple CNN model for bean classification
class BeanClassifierCNN(nn.Module):
    def __init__(self, num_classes=3):
        """
        Initialize the CNN with convolutional and fully connected layers.
        """
        super(BeanClassifierCNN, self).__init__()

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization for conv2
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)  # Batch normalization for conv3
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 128)  # Adjusted to match output size after third conv layer
        # Dropout layer
        
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for classification



    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Apply convolutional layers, batch normalization, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # First conv layer + BatchNorm + ReLU + Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Second conv layer + BatchNorm + ReLU + Pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Third conv layer + BatchNorm + ReLU + Pooling

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 8 * 8)

        # Apply fully connected layers with dropout
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)  # Apply dropout after fc1 (before fc2)
        x = self.fc2(x)  # Output logits

        return x