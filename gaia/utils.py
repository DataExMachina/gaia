"""Utils module."""

import numpy as np


def spatial_weighting(Z, new_coord, pairwise_distance):
    """Initialization of SpatialModel.

    Parameters
    ----------
    Z : {array-like, sparse matrix} of shape (n_samples_1, n_coordinates)
        Training spatial coordinates, where `n_samples_1` is the number of samples and
        `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.
    new_coord : {array-like, sparse matrix} of shape (n_samples_2, n_coordinates)

    Returns
    -------
    weights: np.ndarray of shape(n_samples_1, n_samples_2)
        Weights associated to each Z points according their
        pairwise distance and distance to new_coord.
    """
    distance_matrix = pairwise_distance(Z, Z)
    new_to_Z = pairwise_distance(Z, new_coord)
    weights = np.linalg.inv(distance_matrix).dot(new_to_Z)
    return weights
