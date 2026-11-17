import torch

from plasticity.config import RunConfig
from plasticity.data import (COLORS, build_test_sets, colorize, fit_gaussian_clone,
                             gpu_augment_cifar, independent_sample, sample_color_idx)


def test_colorize_uses_class_color():
    gray = torch.ones(2, 1, 32, 32)
    out = colorize(gray, torch.tensor([0, 5]))
    assert out.shape == (2, 3, 32, 32)
    assert torch.allclose(out[1, :, 0, 0], COLORS[5])


def test_sample_color_idx_correlation():
    y = torch.arange(10).repeat(1000)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample_color_idx(y, 1.0, g), y)
    frac = (sample_color_idx(y, 0.0, g) == y).float().mean()
    assert 0.07 < frac < 0.13  # chance level ~ 0.1


def test_gaussian_clone_matches_class_means(fake_raw):
    x = fake_raw("cifar")["x_train"].float() / 255
    y = fake_raw("cifar")["y_train"]
    xg, yg = fit_gaussian_clone(x, y, n=2000, seed=0)
    assert xg.shape == (2000, 3, 32, 32) and xg.min() >= 0 and xg.max() <= 1
    m_real = x[y == 3].mean(0)
    m_fake = xg[yg == 3].mean(0)
    assert (m_real - m_fake).abs().mean() < 0.05


def test_gaussian_clone_handles_constant_pixels():
    x = torch.zeros(100, 3, 32, 32)
    x[:, :, 10:20, 10:20] = torch.rand(100, 3, 10, 10)
    xg, _ = fit_gaussian_clone(x, torch.arange(10).repeat(10), n=50, seed=0)
    assert torch.isfinite(xg).all()


def test_independent_sample_copies_same_class_pixels(fake_raw):
    raw = fake_raw("cifar")
    x = raw["x_train"].float() / 255
    y = raw["y_train"]
    xi, yi = independent_sample(x, y, n=60, seed=0)
    assert xi.shape == (60, 3, 32, 32) and yi.shape == (60,)
    for i in range(60):
        cls = x[y == yi[i]]  # (m,3,32,32)
        found = (cls == xi[i]).any(0)  # some same-class image matches at each position
        assert found.all()


def test_test_sets_cached_and_identical(tmp_path, fake_raw):
    cfg = RunConfig("cifar_fresh", root=str(tmp_path), eval_size=100)
    a = build_test_sets(cfg, fake_raw("cifar"))
    b = build_test_sets(cfg, fake_raw("cifar"))
    assert set(a) == {"real", "gaussian", "independent"}
    assert torch.equal(a["gaussian"].tensors[0], b["gaussian"].tensors[0])
    assert any((tmp_path / "data" / "testsets").iterdir())


def test_cmnist_test_sets(tmp_path, fake_raw):
    sets = build_test_sets(RunConfig("cmnist_phase1", root=str(tmp_path), eval_size=100), fake_raw("mnist"))
    assert set(sets) == {"color_matched", "color_random", "color_conflict", "gaussian"}
    x, ys, yc = sets["color_conflict"].tensors
    assert x.shape[1:] == (3, 32, 32) and (ys != yc).all()


def test_cmnist_phases_share_cache(tmp_path, fake_raw):
    build_test_sets(RunConfig("cmnist_phase1", root=str(tmp_path), eval_size=100), fake_raw("mnist"))
    files = sorted((tmp_path / "data" / "testsets").iterdir())
    build_test_sets(RunConfig("cmnist_phase2", branch_n=1, root=str(tmp_path), eval_size=100), fake_raw("mnist"))
    assert sorted((tmp_path / "data" / "testsets").iterdir()) == files


def test_gpu_augment_keeps_shape_and_range():
    x = torch.rand(8, 3, 32, 32)
    out = gpu_augment_cifar(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
