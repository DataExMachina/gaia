"""Utils module."""

import numpy as np


def spatial_weighting(centroids, new_coords, pairwise_distance):
    """Initialization of SpatialModel.

    Parameters
    ----------
    centroids : {array-like, sparse matrix} of shape (n_samples_1, n_coordinates)
        Training spatial coordinates, where `n_samples_1` is the number of clusters and
        `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.
    new_coord : {array-like, sparse matrix} of shape (n_samples_2, n_coordinates)

    Returns
    -------
    weights: np.ndarray of shape(n_samples_1, n_samples_2)
        Weights associated to each Z points according their
        pairwise distance and distance to new_coord.
    """
    inner_dist = pairwise_distance(centroids)
    outer_dist = pairwise_distance(centroids, new_coords)
    weights = np.linalg.inv(inner_dist).dot(outer_dist)
    return weights
