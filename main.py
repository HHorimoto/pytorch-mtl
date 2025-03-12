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

from src.data.dataset import create_dataset
from src.utils.seeds import fix_seed, worker_init_fn
from src.models.models import Net
from src.models.coachs import Coach
from src.models.evaluate import evaluate

def main():
    epochs = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataset(root='./data/', download=True, batch_size=64)

    model = Net(input_channel=3, num_class=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    coach = Coach(model, train_loader, test_loader, loss_fn, optimizer, device, epochs)
    coach.train_test()
    trues, preds = coach.evaluate()
    accuracy = accuracy_score(trues, preds)

    print("accuracy: ", accuracy)

if __name__ == "__main__":
    fix_seed()
    main()