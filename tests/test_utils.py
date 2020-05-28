"""Tests for `gaia.utils` package."""

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_distances,
    euclidean_distances,
    haversine_distances,
    manhattan_distances,
    rbf_kernel,
)
from gaia.utils import spatial_weighting

DISTANCES = [euclidean_distances, haversine_distances, manhattan_distances, rbf_kernel]


def test_spatial_weighting():
    centroids = np.random.rand(10, 2)
    new_coords = np.random.rand(500, 2)

    for pairwise_distance in DISTANCES:
        weighted_matrix = spatial_weighting(centroids, new_coords, pairwise_distance)
        np.testing.assert_allclose(
            pairwise_distance(centroids).dot(weighted_matrix),
            pairwise_distance(centroids, new_coords),
            rtol=1e-5,
        )
