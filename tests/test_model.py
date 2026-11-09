import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from hssnb.models import HSSNB


def test_default_input_gives_class_scores():
    model = HSSNB(num_classes=16)
    with torch.no_grad():
        out = model(torch.randn(2, 1, 30, 25, 25))
    assert out.shape == (2, 16)


@pytest.mark.parametrize(
    ("n_components", "window_size"), [(13, 11), (15, 13), (40, 27)], ids=["min", "small", "large"]
)
def test_dw3_layer_sizes_follow_component_count(n_components, window_size):
    model = HSSNB(num_classes=5, n_components=n_components)
    with torch.no_grad():
        out = model(torch.randn(2, 1, n_components, window_size, window_size))
    assert out.shape == (2, 5)


@given(n_components=st.integers(-5, 12))
def test_dw3_component_count_below_13_rejected(n_components):
    with pytest.raises(ValueError, match="n_components"):
        HSSNB(num_classes=4, n_components=n_components)


@given(n_components=st.integers(13, 40))
@settings(max_examples=28)
def test_dw3_component_count_from_13_sizes_the_layers(n_components):
    model = HSSNB(num_classes=4, n_components=n_components)
    assert model.conv2d_1.conv.in_channels == 32 * (n_components - 12)
    with torch.no_grad():
        assert model(torch.randn(1, 1, n_components, 11, 11)).shape == (1, 4)
