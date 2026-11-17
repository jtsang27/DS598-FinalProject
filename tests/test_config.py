from dataclasses import replace
from pathlib import Path

import pytest

from plasticity.config import RunConfig, load_config, run_dirs, run_id


def test_load_config_applies_overrides(tmp_path):
    p = tmp_path / "c.json"; p.write_text('{"experiment": "cifar_fresh", "lr": 0.1}')
    cfg = load_config(str(p), {"seed": 3, "lr": None})
    assert cfg.lr == 0.1 and cfg.seed == 3 and cfg.batch_size == 128


def test_load_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.json"; p.write_text('{"experiment": "cifar_fresh", "lrr": 0.1}')
    with pytest.raises(ValueError, match="lrr"):
        load_config(str(p), {})


def test_run_id_formats():
    assert run_id(RunConfig("cifar_fresh")) == "resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0"
    assert run_id(RunConfig("cmnist_phase2", branch_n=4)).endswith("_color1.00_N4ep")


def test_run_id_changes_with_every_setting():
    base = RunConfig("cmnist_phase2", branch_n=2)
    for field, value in [("alpha", 2.0), ("lr", 0.1), ("weight_decay", 0.0),
                         ("batch_size", 64), ("seed", 1), ("color_corr", 0.9), ("branch_n", 4)]:
        assert run_id(replace(base, **{field: value})) != run_id(base), field


def test_run_id_ignores_total_steps():
    assert run_id(RunConfig("cifar_fresh", total_steps=10)) == run_id(RunConfig("cifar_fresh"))


def test_run_dirs_share_cmnist_folder():
    a = run_dirs(RunConfig("cmnist_phase1", root="/r"))
    b = run_dirs(RunConfig("cmnist_phase2", branch_n=1, root="/r"))
    assert a["ckpt"].parent == b["ckpt"].parent == Path("/r/checkpoints/cmnist")
    assert a["figures"] == Path("/r/figures/cmnist") and a["data"] == Path("/r/data")


@pytest.mark.parametrize("name", ["cifar_fresh", "cmnist_phase1", "cmnist_phase2"])
def test_shipped_configs_load(name):
    cfg = load_config(str(Path(__file__).parents[1] / "configs" / f"{name}.json"), {})
    assert cfg.experiment == name


def test_run_id_phase2_requires_branch_n():
    with pytest.raises(ValueError, match="branch_n"):
        run_id(RunConfig("cmnist_phase2"))
    assert run_id(RunConfig("cmnist_phase1")).endswith("_color1.00")


def test_run_id_independent_of_numeric_type():
    assert run_id(RunConfig("cifar_fresh", alpha=1, weight_decay=0, lr=1)) == \
        run_id(RunConfig("cifar_fresh", alpha=1.0, weight_decay=0.0, lr=1.0))
    assert RunConfig("cifar_fresh", alpha=1).alpha == 1.0
    assert isinstance(RunConfig("cifar_fresh", batch_size=128.0).batch_size, int)
