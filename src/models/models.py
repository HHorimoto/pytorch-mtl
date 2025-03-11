import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 2 * 64,  100)
        self.fc2 = nn.Linear(100, 24)
        self.fc3 = nn.Linear(24, classes)
    
    def forward(self, x):
        h = self.pool(self.relu(self.conv1(x)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.pool(self.relu(self.conv3(h)))
        h = self.relu(self.conv4(h))
        h = h.view(h.size()[0], -1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.fc3(h)
        return h