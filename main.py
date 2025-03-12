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
from src.models.models import Net, MLTNet
from src.models.coachs import Coach, MTLCoach
from src.visualization.visualize import plot

def main():

    with open('config.yaml') as file:
        config_file = yaml.safe_load(file)
    print(config_file)

    ROOT = config_file['config']['root']
    EPOCHS = config_file['config']['epochs']
    BATCH_SIZE = config_file['config']['batch_size']
    LR = config_file['config']['learning_rate']
    LEARNING = config_file['config']['learning']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = create_dataset(root=ROOT, download=True, batch_size=BATCH_SIZE)

    if LEARNING == 'STL':
        print('single task learning')
        model = Net(input_channel=3, num_class=10).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        coach = Coach(model, train_loader, test_loader, loss_fn, optimizer, device, EPOCHS)

    elif LEARNING == 'MTL':
        print('multi task learning')
        model = MLTNet(input_channel=3, num_class=[10, 2]).to(device)
        loss_fn1, loss_fn2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
        lambda1, lambda2 = 1., 1.
        optimizer = optim.Adam(model.parameters(), lr=LR)
        coach = MTLCoach(model, train_loader, test_loader, loss_fn1, loss_fn2,
                         lambda1, lambda2, optimizer, device, EPOCHS)

    else:
        print(LEARNING)
        sys.exit()

    coach.train_test()
    train_loss, test_loss = coach.train_loss, coach.test_loss
    trues, preds = coach.evaluate()
    accuracy = accuracy_score(trues, preds)

    print("accuracy: ", accuracy)
    print(trues, preds)

    plot(train_loss, test_loss)

if __name__ == "__main__":
    fix_seed()
    main()