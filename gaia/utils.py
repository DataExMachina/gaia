"""Utils module."""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def spatial_weighting(Z, new_coord):
    """Initialization of SpatialModel.

    Parameters
    ----------
    Z : {array-like, sparse matrix} of shape (n_samples, n_coordinates)
        Training spatial coordinates, where `n_samples` is the number of samples and
        `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.
    new_coord : {array-like, sparse matrix} of shape (1, n_coordinates)

    Returns
    -------
    weights: np.ndarray
        Weights associated to each Z points according their
        pairwise distance and distance to new_coord.
    """
    distance_matrix = euclidean_distances(Z, Z)
    new_to_Z = euclidean_distances(Z, new_coord)
    weights = np.linalg.inv(distance_matrix).dot(new_to_Z)
    return weights
