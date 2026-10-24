"""FedCurv (Shoham et al., 2019) as a Flower 1.42 strategy."""
from collections.abc import Iterable

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

KEY = "fedcurv"


def pack(F: np.ndarray, FT: np.ndarray) -> ConfigRecord:
    """Two float32 vectors -> ConfigRecord of bytes."""
    return ConfigRecord({"F": F.astype(np.float32).tobytes(),
                         "FT": FT.astype(np.float32).tobytes()})


def unpack(rec: ConfigRecord):
    return (np.frombuffer(rec["F"], dtype=np.float32).copy(),
            np.frombuffer(rec["FT"], dtype=np.float32).copy())


class FedCurv(FedAvg):
    def __init__(self, curv_lambda: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.curv_lambda = curv_lambda
        self.stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # node_id -> (F, F*theta)

    def configure_train(self, server_round: int, arrays: ArrayRecord,
                        config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        config["curv-lambda"] = self.curv_lambda
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if not self.stats:                       # round 1: nothing to send yet
            return messages
        tot_F = sum(F for F, _ in self.stats.values())
        tot_FT = sum(FT for _, FT in self.stats.values())
        out = []
        for msg in messages:                     # one message per hospital
            nid = msg.metadata.dst_node_id
            F, FT = self.stats.get(nid, (0.0, 0.0))
            content = RecordDict({k: v for k, v in msg.content.items()})
            content[KEY] = pack(tot_F - F, tot_FT - FT)   # sums over the OTHER hospitals
            out.append(Message(content=content, message_type=MessageType.TRAIN,
                               dst_node_id=nid))
        return out

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        for msg in replies:
            if msg.has_error() or KEY not in msg.content:
                continue
            self.stats[msg.metadata.src_node_id] = unpack(msg.content[KEY])
        return super().aggregate_train(server_round, replies)