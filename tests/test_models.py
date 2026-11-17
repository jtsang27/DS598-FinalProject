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
