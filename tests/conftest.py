import pytest
import torch


def _make_raw(kind: str) -> dict:
    g = torch.Generator().manual_seed(0)
    n_train, n_test = 200, 100
    if kind == "cifar":
        shape = (3, 32, 32)
        mk = lambda n: torch.randint(0, 256, (n, *shape), generator=g, dtype=torch.uint8)
    elif kind == "mnist":
        def mk(n):
            x = torch.zeros(n, 1, 32, 32, dtype=torch.uint8)
            for i in range(n):
                r, c = torch.randint(4, 20, (2,), generator=g).tolist()
                x[i, 0, r:r + 8, c:c + 8] = torch.randint(100, 256, (8, 8), generator=g, dtype=torch.uint8)
            return x
    else:
        raise ValueError(kind)
    return {
        "x_train": mk(n_train), "y_train": torch.arange(n_train) % 10,
        "x_test": mk(n_test), "y_test": torch.arange(n_test) % 10,
    }


@pytest.fixture
def fake_raw():
    """Callable fake_raw(kind) -> deterministic raw dict; kind in {"cifar", "mnist"}."""
    return _make_raw
