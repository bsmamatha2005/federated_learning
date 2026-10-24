import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federated_learning_app.fedcurv import pack, unpack
from federated_learning_app.task import Net, fisher_diag, flat_params, load_data, train

torch.manual_seed(0)
model = Net()
trainloader, testloader = load_data(0, 32)

print("1) normal training")
print("   loss:", train(model, trainloader, 1, 0.01, "cpu"))

print("2) Fisher calculation")
F = fisher_diag(model, trainloader, "cpu")
theta = flat_params(model)
print("   Fisher size:", F.shape, "| weights size:", theta.shape)

print("3) packing and unpacking")
A, b = unpack(pack(F, F * theta))
print("   OK:", A.shape, b.shape)

print("4) training with the FedCurv penalty")
print("   loss:", train(model, trainloader, 1, 0.01, "cpu", 0.0, (A, b), 1.0))

print("ALL OK")