"""Main module."""

import numpy as np
from rtree import index
from scipy import spatial
from sklearn.base import is_classifier, is_regressor

from gaia.utils import spatial_weighting


class SpatialModel:
    """Base class for spatial modeling."""

    def __init__(self, estimator, n_neighbors=20):
        """Initialization of SpatialModel.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
            A sklearn model with fit / predict pradigm.
        n_neighbors : int
            Number of neighors to considere for each new point.
        """
        if is_classifier(estimator) or is_regressor(estimator):
            self.estimator = estimator
        else:
            raise TypeError(
                "It does not seem to be a scikit-learn estimator "
                "for supervised learning."
            )

        if n_neighbors > 2 and isinstance(n_neighbors, int):
            self.n_neighbors = n_neighbors
        else:
            raise ValueError(
                "Number of neighbors (%s) is invalid:"
                "It has to be a integer higher than 2."
            )

    def fit(self, X, Z, y):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Z : {array-like, sparse matrix} of shape (n_samples, n_coordinates)
            Training spatial coordinates, where `n_samples` is the number of samples and
            `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """
        self.spatial_index = spatial.KDTree(Z)
        self.train_X = X
        self.train_Z = Z
        self.train_y = y

    def predict(self, X, Z):
        """yolo"""
        self.models = list()
        results = list()
        for line in range(X.shape[0]):

            # get sub data
            index = self.spatial_index.query(Z[line, :], k=self.n_neighbors)[1]
            local_train_Z = self.train_Z[index, :]
            local_train_X = self.train_X[index, :]
            local_train_y = self.train_y[index]

            weights = spatial_weighting(local_train_Z, Z[line, :])
            self.models.append(
                self.estimator.fit(local_train_X, local_train_y, weights).copy()
            )
            results = self.models[-1].predict(X[line, :])
        return results
