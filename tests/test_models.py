import torch
import torch.nn as nn

from plasticity.models import resnet18_cifar


def test_output_shape():
    assert resnet18_cifar()(torch.rand(4, 3, 32, 32)).shape == (4, 10)


def test_alpha_scales_weights():
    torch.manual_seed(0); a = resnet18_cifar(alpha=1.0)
    torch.manual_seed(0); b = resnet18_cifar(alpha=2.0)
    assert torch.allclose(b.fc.weight, 2 * a.fc.weight)
    assert torch.allclose(b.conv1.weight, 2 * a.conv1.weight)
    assert torch.allclose(b.bn1.weight, a.bn1.weight)      # BatchNorm untouched


def test_stem_is_cifar_style():
    m = resnet18_cifar()
    assert m.conv1.kernel_size == (3, 3) and m.conv1.stride == (1, 1)
    assert isinstance(m.maxpool, nn.Identity)


def test_stem_uses_kaiming_fan_out_init():
    torch.manual_seed(0)
    w = resnet18_cifar().conv1.weight
    fan_out = w.shape[0] * w.shape[2] * w.shape[3]     # 64 * 3 * 3
    expected = (2.0 / fan_out) ** 0.5                   # ~0.059; PyTorch's default would be ~0.11
    assert abs(w.std().item() / expected - 1) < 0.1
    assert abs(w.mean().item()) < 0.2 * expected
    torch.manual_seed(0)
    assert torch.allclose(resnet18_cifar(alpha=0.5).conv1.weight, 0.5 * w)
