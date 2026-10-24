import torch
import numpy as np
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


def flat_params(model) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def train(model, loader, epochs, lr, device, mu=0.0, curv=None, lam=0.0):
    model.to(device).train()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    global_params = [p.detach().clone() for p in model.parameters()]  # server's model
    if curv is not None:                                  # FedCurv: A = sum F_j, b = sum F_j*theta_j
        A = torch.from_numpy(curv[0]).to(device)
        b = torch.from_numpy(curv[1]).to(device)
    total, n = 0.0, 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            base = crit(model(X), y)
            loss = base
            if mu > 0:                                    # FedProx
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2) * prox
            if curv is not None and lam > 0:              # FedCurv
                theta = torch.cat([p.reshape(-1) for p in model.parameters()])
                loss = loss + lam * ((A * theta ** 2).sum() - 2 * (b * theta).sum())
            loss.backward()
            opt.step()
            total += base.item() * len(y)
            n += len(y)
    return total / n


def fisher_diag(model, loader, device) -> np.ndarray:
    """Diagonal empirical Fisher: mean of squared per-sample gradients."""
    model.to(device).eval()
    crit = nn.BCEWithLogitsLoss()
    F = [torch.zeros_like(p) for p in model.parameters()]
    n = 0
    for X, y in loader:
        for i in range(len(y)):
            model.zero_grad()
            crit(model(X[i:i + 1].to(device)), y[i:i + 1].to(device)).backward()
            for f, p in zip(F, model.parameters()):
                f += p.grad.detach() ** 2
            n += 1
    return (torch.cat([f.reshape(-1) for f in F]).cpu().numpy() / n).astype(np.float32)

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