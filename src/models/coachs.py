import torch
import torch.nn as nn

import numpy as np

class Coach:
    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, device, epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        # store
        self.train_loss = []
        self.test_loss = []

    def _train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        train_batch_loss = []
        for batch, (X, y, _) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            output = self.model(X)
            loss = self.loss_fn(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_batch_loss.append(loss.item())
        return np.mean(train_batch_loss)
            
    def _test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        test_batch_loss = []
        with torch.no_grad():
            for X, y, _ in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output, y)

                test_batch_loss.append(loss.item())
        return np.mean(test_batch_loss)

    def train_test(self):
        for epoch in range(self.epochs):
            train_epoch_loss = self._train_epoch()
            test_epoch_loss = self._test_epoch()

            print("epoch: ", epoch+1, "/", self.epochs)
            print("train loss: ", train_epoch_loss)
            print("test loss: ", test_epoch_loss)

            self.train_loss.append(train_epoch_loss)
            self.test_loss.append(test_epoch_loss)

    def evaluate(self):
        self.model.eval()
        preds, tures = [], []

        dataloader = self.test_loader
        with torch.no_grad():
            for X, y, _ in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                tures.append(y)
                preds.append(output)
                
        tures = torch.cat(tures, axis=0)
        preds = torch.cat(preds, axis=0)
        _, preds = torch.max(preds, 1)
        

        tures = tures.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
                
        return tures, preds
    
class MTLCoach(Coach):
    def __init__(self, model, train_loader, test_loader, 
                 loss_fn1, loss_fn2, lambda1, lambda2, optimizer, device, epochs):
        super().__init__(model, train_loader, test_loader, loss_fn1, optimizer, device, epochs)
        
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def _train_epoch(self):
        self.model.train()
        dataloader = self.train_loader
        train_batch_loss = []
        for batch, (X, y1, y2) in enumerate(dataloader):
            X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)

            output1, output2 = self.model(X)
            loss1, loss2 = self.loss_fn1(output1, y1), self.loss_fn2(output2, y2)
            loss = self.lambda1 * loss1 + self.lambda2 * loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_batch_loss.append(loss.item())
        return np.mean(train_batch_loss)
    
    def _test_epoch(self):
        self.model.eval()
        dataloader = self.test_loader
        test_batch_loss = []
        with torch.no_grad():
            for X, y1, y2 in dataloader:
                X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)

                output1, output2 = self.model(X)
                loss1, loss2 = self.loss_fn1(output1, y1), self.loss_fn2(output2, y2)
                loss = self.lambda1 * loss1 + self.lambda2 * loss2
                
                test_batch_loss.append(loss.item())
        return np.mean(test_batch_loss)
    
    def evaluate(self):
        self.model.eval()
        tures, preds = [], []

        dataloader = self.test_loader
        with torch.no_grad():
            for X, y1, y2 in dataloader:
                X, y1, y2 = X.to(self.device), y1.to(self.device), y2.to(self.device)

                output1, output2 = self.model(X)
                tures.append(y1)
                preds.append(output1)

        tures = torch.cat(tures, axis=0)
        preds = torch.cat(preds, axis=0)
        _, preds = torch.max(preds, 1)

        tures = tures.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        return tures, preds
    
    def train_test(self):
        return super().train_test()