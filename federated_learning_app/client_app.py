import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .fedcurv import KEY, pack, unpack
from .task import Net, fisher_diag, flat_params, load_data
from .task import test as test_fn
from .task import train as train_fn

app = ClientApp()
DEVICE = torch.device("cpu")


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    idx = context.node_config["partition-id"]          # 0..3 = which hospital
    trainloader, _ = load_data(idx, context.run_config["batch-size"])
    cfg = msg.content["config"]
    mu = float(cfg.get("proximal-mu", 0.0))
    lam = float(cfg.get("curv-lambda", 0.0))
    curv = unpack(msg.content[KEY]) if KEY in msg.content else None
    loss = train_fn(model, trainloader, context.run_config["local-epochs"],
                    cfg["lr"], DEVICE, mu, curv, lam)
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss,
                                 "num-examples": len(trainloader.dataset)}),
    })
    if lam > 0:                      # FedCurv: also send Fisher information
        F = fisher_diag(model, trainloader, DEVICE)
        content[KEY] = pack(F, F * flat_params(model))
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    idx = context.node_config["partition-id"]
    _, testloader = load_data(idx, context.run_config["batch-size"])
    loss, acc = test_fn(model, testloader, DEVICE)
    content = RecordDict({"metrics": MetricRecord({
        "eval_loss": loss, "eval_acc": acc,
        "num-examples": len(testloader.dataset)})})
    return Message(content=content, reply_to=msg)