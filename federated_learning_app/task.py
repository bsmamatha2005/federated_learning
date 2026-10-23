import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import INPUT_DIM, load_hospital


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # raw logit; sigmoid applied only at prediction time


def load_data(idx: int, batch_size: int):
    Xtr, Xte, ytr, yte = load_hospital(idx)
    train = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size))


def train(model, loader, epochs, lr, device):
    model.to(device).train()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    total, n = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
    return total / n


def test(model, loader, device):
    model.to(device).eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss += crit(out, y).item()
            correct += ((out > 0).float() == y).sum().item()
            n += len(y)
    return loss / n, correct / n