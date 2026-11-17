import torch
import torch.nn.functional as F

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


def test_gpu_augment_is_crop_of_padded_maybe_flipped():
    x = torch.rand(32, 3, 32, 32) + 0.5  # nonzero so zero padding is distinguishable
    out = gpu_augment_cifar(x)
    padded = F.pad(x, (4, 4, 4, 4))
    for i in range(len(x)):
        crops = [padded[i, :, dy:dy + 32, dx:dx + 32] for dy in range(9) for dx in range(9)]
        assert any(torch.equal(c, out[i]) or torch.equal(c.flip(2), out[i]) for c in crops)


def test_gaussian_clone_reproduces_covariance():
    g = torch.Generator().manual_seed(0)
    k, n = 3, 4000
    mix = torch.randn(k, 3 * 32 * 32, generator=g)
    z = torch.randn(n, k, generator=g)
    flat = z @ mix * 0.06 + 0.5  # rank-3 correlated pixels, std ~0.1 around 0.5 (clipping negligible)
    x = flat.reshape(n, 3, 32, 32)
    y = torch.zeros(n, dtype=torch.long)
    xg, _ = fit_gaussian_clone(x, y, n=n, seed=1)
    a, b = x.reshape(n, -1), xg.reshape(n, -1)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    # pick the most strongly correlated pixel pair (off-diagonal) in the real data
    corr = ca / torch.sqrt(torch.diag(ca)[:, None] * torch.diag(ca)[None, :])
    corr.fill_diagonal_(0)
    i, j = divmod(corr.abs().argmax().item(), corr.shape[1])
    assert corr[i, j].abs() > 0.5 and abs(ca[i, j]) > 1e-3
    assert abs(cb[i, j] - ca[i, j]) < 0.25 * abs(ca[i, j])
    # a diagonal-only fit would give ~0 here; clone must be nowhere near that
    assert (cb - ca).norm() < 0.3 * ca.norm()
    assert (torch.diag_embed(torch.diag(ca)) - ca).norm() > 0.5 * ca.norm()


def test_cache_key_includes_train_subset(tmp_path, fake_raw):
    for sub in (None, 100):
        cfg = RunConfig("cifar_fresh", root=str(tmp_path), eval_size=50, train_subset=sub)
        build_test_sets(cfg, fake_raw("cifar"))
    files = list((tmp_path / "data" / "testsets").glob("gaussian_*.pt"))
    assert len(files) == 2
