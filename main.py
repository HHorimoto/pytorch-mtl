import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from PIL import Image

from sklearn.metrics import accuracy_score

from src.utils.seeds import fix_seed
from src.data.mtfl_dataset import create_dataset
from src.models.models import MultiCNN
from src.models.coachs import MultiCoach

def main():
    DATA_PATH = "./data/mtfl"
    EPOCHS = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = create_dataset(data_path=DATA_PATH, batch_size=64)

    model = MultiCNN().to(device)

    loss_fn1, loss_fn2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    lambda1, lambda2 = 1., 1.
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_loss, test_loss = [], []
    coach = MultiCoach(model, train_loader, test_loader, 
                       loss_fn1, loss_fn2, lambda1, lambda2, optimizer, device)
    for epoch in range(EPOCHS):
        coach.train_epoch()
        coach.test_epoch()
        
        print("epoch: ", epoch+1, "/", EPOCHS)
        train_epoch_loss, test_epoch_loss = coach.train_epoch_loss, coach.test_epoch_loss
        print("train loss: ", train_epoch_loss)
        print("test loss: ", test_epoch_loss)
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)

if __name__ == "__main__":
    fix_seed()
    main()