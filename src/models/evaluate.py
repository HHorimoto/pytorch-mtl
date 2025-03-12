import torch

def evaluate(model, dataloader, device):
    model.eval()
    preds, tures = [], []

    with torch.no_grad():
        for X, y, _ in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            tures.append(y)
            preds.append(output)

    tures = torch.cat(tures, axis=0)
    preds = torch.cat(preds, axis=0)
    _, preds = torch.max(preds, 1)

    tures = tures.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    return tures, preds