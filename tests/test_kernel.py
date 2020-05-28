"""Tests for `gaia.kernel` package."""

import numpy as np
import pytest
from gaia.kernel import Kernel

@pytest.fixture
def ker_class():
    """Initialize a Kernel class"""
    ker = Kernel()
    return ker

def test_kernel_method_raises_exception_on_bad_kernel_name():
    with pytest.raises(ValueError):
        Kernel(method="unknown_kernel_name")

def test_kernel_method_raises_exception_on_bad_bandwidth_value():
    with pytest.raises(ValueError):
        Kernel(bandwidth=0)
    with pytest.raises(ValueError):
        Kernel(bandwidth=1.1)
    with pytest.raises(ValueError):
        Kernel(bandwidth="a_bad_string_value")

def test_kernel_norm(ker_class):
    vector = np.random.uniform(-10, 10, (1, 50))
    bandwidths = np.arange(0.1, 1, 0.1)

    for bandwidth in bandwidths:
        normed_vector = ker_class._kernel_norm(vector, bandwidth)
        assert normed_vector.min()>=0
        assert normed_vector.max()<=1

def test_kernel_check_nan(ker_class):
    vector = np.random.rand(50)

    vector_with_nan = np.random.rand(50)
    vector_with_nan.ravel()[np.random.choice(vector_with_nan.size, 5, replace=False)] = np.nan

    assert not ker_class._check_nan(vector)
    assert ker_class._check_nan(vector_with_nan)

def test_iter_bandwidth_search(ker_class):
    vector = np.random.uniform(-10, 10, (5, 500))
    auto_bandwidth = ker_class._iter_bandwidth_search(vector)
    assert auto_bandwidth > 0 and auto_bandwidth <=1

def test_kernel(ker_class):
    vector = np.random.uniform(-10, 10, (5, 500))
    normed_vector = ker_class.kernel(vector)
    assert normed_vector.shape