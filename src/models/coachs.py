import torch
import torch.nn as nn

import numpy as np

class MultiCoach:
    def __init__(self, model, train_loader, test_loader, 
                 loss_fn1, loss_fn2, lambda1, lambda2, optimizer, device):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.optimizer = optimizer
        self.device = device

        # store
        self.train_epoch_loss = []
        self.test_epoch_loss = []

    def train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        train_batch_loss = []
        for batch, (X, y1, y2) in enumerate(dataloader):
            X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)

            output = self.model(X)
            loss1 = self.loss_fn1(output[:, :2], y1)
            loss2 = self.loss_fn2(output[:, 2:], y2)
            loss = self.lambda1 * loss1 + self.lambda2 * loss2
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_batch_loss.append(loss.item())
        self.train_epoch_loss = np.mean(train_batch_loss)
            
    def test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        test_batch_loss = []
        with torch.no_grad():
            for X, y1, y2 in dataloader:
                X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)
                output = self.model(X)
                loss1 = self.loss_fn1(output[:, :2], y1)
                loss2 = self.loss_fn2(output[:, 2:], y2)
                loss = self.lambda1 * loss1 + self.lambda2 * loss2
                
                test_batch_loss.append(loss.item())
        self.test_epoch_loss = np.mean(test_batch_loss)
        