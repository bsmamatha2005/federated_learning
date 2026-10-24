import json

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from .data import ARTIFACT_DIR
from .fedcurv import FedCurv
from .task import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    mu = float(context.run_config["proximal-mu"])
    lam = float(context.run_config["curv-lambda"])

    if mu > 0 and lam > 0:
        raise ValueError("Set only one of proximal-mu (FedProx) or curv-lambda (FedCurv).")
    if lam > 0:
        name = "fedcurv"
        strategy = FedCurv(curv_lambda=lam, fraction_evaluate=1.0)
    elif mu > 0:
        name = "fedprox"
        strategy = FedProx(fraction_evaluate=1.0, proximal_mu=mu)
    else:
        name = "fedavg"
        strategy = FedAvg(fraction_evaluate=1.0)

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(Net().state_dict()),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    torch.save(result.arrays.to_torch_state_dict(), ARTIFACT_DIR / f"{name}_model.pt")

    history = {"algorithm": name, "mu": mu, "curv_lambda": lam}
    for key in ("train_metrics_clientapp", "evaluate_metrics_clientapp"):
        data = getattr(result, key, None)
        if data:
            history[key] = {str(r): dict(m) for r, m in data.items()}
    with open(ARTIFACT_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved artifacts/{name}_model.pt and {name}_history.json")