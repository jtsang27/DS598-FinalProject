import pytest

from plasticity.schedule import check_steps, full_state_steps, log_steps, save_steps


def test_log_steps():
    assert log_steps(10) == [1, 2, 4, 8]
    assert log_steps(0) == []


def test_check_steps_cover_log_half_epoch_and_end():
    s = check_steps(steps_per_epoch=10, total=40)
    assert s[:5] == [0, 1, 2, 4, 5] and 32 in s and s[-1] == 40 and s == sorted(set(s))


def test_dense_vs_sparse_phase1():
    dense = save_steps("cmnist_phase1", "dense", 10, 160, [0, 1, 2, 4, 8, 16])
    sparse = save_steps("cmnist_phase1", "sparse", 10, 160, [0, 1, 2, 4, 8, 16])
    assert {0, 10, 20, 40, 80, 160} <= set(sparse) <= set(dense)
    assert len(sparse) == 6


def test_full_state_at_durations():
    assert full_state_steps("cmnist_phase1", 10, 160, [0, 1, 4, 16]) == [10, 40, 160]


def test_full_state_other_experiments_only_total():
    assert full_state_steps("cifar_fresh", 10, 160, [1, 4]) == [160]
    assert full_state_steps("cmnist_phase2", 10, 160, [1, 4]) == [160]


def test_cifar_dense_includes_every_8_epochs():
    s = save_steps("cifar_fresh", "dense", 10, 200, [])
    assert {0, 1, 80, 160, 200} <= set(s) and s == sorted(set(s))


def test_phase2_excludes_zero():
    dense = save_steps("cmnist_phase2", "dense", 10, 120, [1, 2])
    sparse = save_steps("cmnist_phase2", "sparse", 10, 120, [1, 2])
    assert 0 not in dense and 50 in dense and 100 in dense and dense[-1] == 120
    assert sparse == [120]


def test_unknown_experiment_or_schedule_raises():
    with pytest.raises(ValueError):
        save_steps("nope", "dense", 10, 100, [])
    with pytest.raises(ValueError):
        save_steps("cifar_fresh", "weird", 10, 100, [])


def test_check_steps_dense_prefix_for_phase2():
    base = check_steps(steps_per_epoch=100, total=500)
    dense = check_steps(steps_per_epoch=100, total=500, dense_until=200, dense_every=16)
    assert set(base) <= set(dense)
    assert set(dense) - set(base) == {16 * k for k in range(1, 13)} - set(base)  # 16..192
    assert 208 not in dense and dense == sorted(set(dense))
    # dense prefix is capped at total
    assert max(check_steps(10, 40, dense_until=1000, dense_every=16)) == 40


def test_phase2_check_steps_every_16_for_two_epochs():
    from plasticity.schedule import phase2_check_steps
    spe, total = 468, 20 * 468
    s = set(phase2_check_steps(spe, total))
    assert {16 * k for k in range(1, 2 * spe // 16 + 1)} <= s
    assert set(check_steps(spe, total)) <= s
    assert 16 * (2 * spe // 16 + 1) not in s
