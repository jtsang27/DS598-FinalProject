import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from plasticity.metrics import MetricsLogger, dist_from_init, evaluate, weight_norm


class AlwaysZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = torch.zeros(x.shape[0], 10) + self.w
        out[:, 0] = 10.0
        return out


def test_evaluate_conflict_set_reports_both_labels():
    x = torch.rand(8, 3, 32, 32)
    conflict = TensorDataset(x, torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long))
    plain = TensorDataset(x, torch.zeros(8, dtype=torch.long))
    res = evaluate(AlwaysZero(), {"c": conflict, "p": plain}, "cpu", batch_size=3)
    assert res["c"]["acc"] == 1.0 and res["c"]["acc_color_label"] == 0.0
    assert res["p"]["acc"] == 1.0 and res["p"]["acc_color_label"] is None
    assert res["c"]["loss"] > 0


def test_evaluate_restores_mode():
    m = AlwaysZero()
    ds = {"p": TensorDataset(torch.rand(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))}
    m.train()
    evaluate(m, ds, "cpu")
    assert m.training
    m.eval()
    evaluate(m, ds, "cpu")
    assert not m.training


def test_weight_norm_and_dist():
    m = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[3.0, 4.0]]))
    assert weight_norm(m) == pytest.approx(5.0)
    init = {"weight": torch.tensor([[0.0, 0.0]])}
    assert dist_from_init(m, init) == pytest.approx(5.0)
    assert dist_from_init(m, {k: v.clone() for k, v in m.state_dict().items()}) == 0.0


def test_logger_resume_last_step(tmp_path):
    lg = MetricsLogger(tmp_path / "m.csv")
    assert lg.last_step() is None
    lg.log([{"global_step": 4, "test_set": "real"}])
    assert MetricsLogger(tmp_path / "m.csv").last_step() == 4
    MetricsLogger(tmp_path / "m.csv").log([{"global_step": 9}])
    lines = open(tmp_path / "m.csv").read().splitlines()
    assert lines[0].split(",")[0] == "run_id" and len(lines) == 3  # one header only
    assert MetricsLogger(tmp_path / "m.csv").last_step() == 9


def test_logger_rejects_unknown_keys(tmp_path):
    lg = MetricsLogger(tmp_path / "m.csv")
    with pytest.raises(ValueError, match="bogus"):
        lg.log([{"global_step": 1, "bogus": 2}])
