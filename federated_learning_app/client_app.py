import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .task import Net, load_data
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
    mu = float(msg.content["config"].get("proximal-mu", 0.0))
    loss = train_fn(model, trainloader, context.run_config["local-epochs"],
                    msg.content["config"]["lr"], DEVICE, mu)
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss,
                                 "num-examples": len(trainloader.dataset)}),
    })
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