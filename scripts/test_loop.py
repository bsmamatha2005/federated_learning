"""Runs 2 rounds of FedCurv with the REAL server strategy and REAL client code,
in one process (no Ray), so any hidden error is printed."""
import sys
import traceback
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.supercore.task_identity import TaskIdentity

TaskIdentity.run_id = 1
TaskIdentity.task_id = 1
TaskIdentity.node_id = 0

import flwr.serverapp.strategy.fedavg as fa
from federated_learning_app.client_app import evaluate as client_eval
from federated_learning_app.client_app import train as client_train
from federated_learning_app.fedcurv import FedCurv
from federated_learning_app.task import Net

NODES = [101, 102, 103, 104]            # node i  <->  hospital i
RUN_CONFIG = {"batch-size": 32, "local-epochs": 3}


class FakeGrid:
    def get_node_ids(self):
        return NODES


def ctx(node_id):
    return Context(run_id=1, node_id=node_id,
                   node_config={"partition-id": NODES.index(node_id), "num-partitions": 4},
                   state=RecordDict(), run_config=RUN_CONFIG)


strategy = FedCurv(curv_lambda=1.0, fraction_evaluate=1.0)
arrays = ArrayRecord(Net().state_dict())

try:
    with mock.patch.object(fa, "sample_nodes", lambda g, m, s: (NODES, NODES)):
        for rnd in (1, 2):
            print(f"\n=== ROUND {rnd} ===")
            msgs = list(strategy.configure_train(rnd, arrays, ConfigRecord({"lr": 0.01}), FakeGrid()))
            replies = []
            for m in msgs:
                nid = m.metadata.dst_node_id
                print(f"hospital {NODES.index(nid)}: training...", end=" ", flush=True)
                replies.append(client_train(m, ctx(nid)))
                print("done")
            new_arrays, metrics = strategy.aggregate_train(rnd, replies)
            print("aggregated:", dict(metrics) if metrics else None)
            arrays = new_arrays
            em = list(strategy.configure_evaluate(rnd, arrays, ConfigRecord({}), FakeGrid()))
            ev = [client_eval(m, ctx(m.metadata.dst_node_id)) for m in em]
            emetrics = strategy.aggregate_evaluate(rnd, ev)
            print("eval:", dict(emetrics) if emetrics else None)
    print("\nLOOP OK")
except Exception:
    print("\n!!! FOUND THE ERROR !!!")
    traceback.print_exc()