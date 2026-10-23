import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .data import ARTIFACT_DIR
from .task import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]

    arrays = ArrayRecord(Net().state_dict())
    strategy = FedAvg(fraction_evaluate=1.0)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    torch.save(result.arrays.to_torch_state_dict(), ARTIFACT_DIR / "fedavg_model.pt")
    print("Saved artifacts/fedavg_model.pt")