"""Kernel module."""

import numpy as np


def epanechnikov(vector, bandwidth):
    ker_val = 3 / 4 * (bandwidth - vector ** 2) * (np.abs(vector) < bandwidth)
    return ker_val


class Kernel:

    supported_methods = ["epanechnikov"]

    def __init__(self, bandwidth="auto", method="epanechnikov"):
        """
        Parameters
        ----------
        bandwidth: float
            A float in (0, 1].
        """
        if isinstance(bandwidth, str) and bandwidth == "auto":
            self.bandwidth = bandwidth
        elif isinstance(bandwidth, float) and bandwidth > 0 and bandwidth <= 1:
            self.bandwidth = bandwidth
        else:
            raise ValueError(
                'Wrong bandwitdh value. It must be str ("auto") or float in (0, 1]'
            )

        if method in self.supported_methods:
            self._kernel_method = eval(method)
        else:
            raise ValueError(
                "Wrong method name, it must be on of %s" % self.supported_methods
            )

    def _kernel_norm(self, vector, bandwidth):
        """Apply kernel method with double normalization."""

        vector = (vector.T / vector.sum(1)).T
        vector = np.abs(self._kernel_method(vector, bandwidth))
        vector = (vector.T / vector.sum(1)).T
        return vector

    def _check_nan(self, vector):
        """Return `True` if there are NaNs in vector"""
        return np.isnan(vector).sum() > 0

    def _iter_bandwidth_search(self, vector):
        """Find minimal valid bandwidth by dichotomic search."""
        bandwidth_list = list(np.arange(1e-6, 1, 0.001))
        nan_init = list(
            map(
                lambda b: self._check_nan(self._kernel_norm(vector, b)), bandwidth_list,
            )
        )
        for nan_index in range(len(nan_init)):
            if np.isnan(nan_init[nan_index]):
                continue
            else:
                return bandwidth_list[nan_index]

    def kernel(self, vector):
        """Generic kernel caller.

        Parameters
        ----------
        vector: {array-like, sparse matrix} of shape (n_samples, n_cols)
                Vectors, where `n_samples` is the number of samples and
                `n_cols` is the number of columns on which normalization
                is done (twice).
        """
        if self.bandwidth == "auto":
            bandwidth = self._iter_bandwidth_search(vector)
        else:
            bandwidth = self.bandwidth

        vector = self._kernel_norm(vector, bandwidth)
        return vector
