"""Main module."""

import copy
import numpy as np
from rtree import index
from scipy import spatial
from sklearn.base import is_classifier, is_regressor

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

from gaia.utils import spatial_weighting


def weighted_average(y, weights):
    """

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
    weights : np.ndarray of shape (n_sample,)

    Returns
    -------
    pred : np.ndarray
    """
    pred = np.sum(y * weights)
    return pred


class SpatialModel:
    """Base class for spatial modeling."""

    def __init__(
        self,
        cluster=KMeans(),
        estimator=LinearRegression(),
        pairwise_distance=euclidean_distances,
    ):
        """Initialization of SpatialModel.

        Parameters
        ----------
        cluster: sklearn.cluster
            Clustering method.
        estimator: sklearn.base.BaseEstimator
            A sklearn model with fit / predict paradigm.
        pairwise_distance: sklearn.metrics.pairwise
            A distance metric
        """
        # define clusteting method
        if (
            (cluster._estimator_type == "clusterer")
            and hasattr(cluster, "predict")
            and hasattr(cluster, "n_clusters")
        ):
            self.cluster = cluster
        else:
            raise TypeError("Wrong method for cluster argument.")

        # declare machine learning method
        if is_classifier(estimator) or is_regressor(estimator):
            self.estimator = estimator
        else:
            raise TypeError(
                "It does not seem to be a scikit-learn estimator "
                "for supervised learning."
            )
        # declare pairwise
        if pairwise_distance.__module__ == "sklearn.metrics.pairwise":
            self.pairwise_distance = pairwise_distance
        else:
            raise TypeError(
                "Wrong pairwise function. You should use functions from sklearn.metrics.pairwise."
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
        # start by running the clustering method
        # and deduce weight (inverse of distance)
        self.cluster.fit(Z)

        # compute spatial weights
        overall_fit_weights = spatial_weighting(
            self.cluster.cluster_centers_, Z, self.pairwise_distance
        )
        overall_fit_weights = overall_fit_weights - overall_fit_weights.min(1).reshape(
            -1, 1
        )
        overall_fit_weights = overall_fit_weights / overall_fit_weights.sum(1).reshape(
            -1, 1
        )

        # iterate over clusters (a model by cluster)
        self.list_estimators = list()
        for num_model in range(self.cluster.n_clusters):
            current_weights = overall_fit_weights[num_model, :]
            current_model = copy.deepcopy(self.estimator)
            current_model.fit(X, y, sample_weight=current_weights)
            self.list_estimators.append(current_model)

    def predict(self, X, Z):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Z : {array-like, sparse matrix} of shape (n_samples, n_coordinates)
            Training spatial coordinates, where `n_samples` is the number of samples and
            `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.

        Returns
        -------
        predicted_values: array, shape (n_samples,)
            Returns predicted values.
        """

        # compute spatial weights
        overall_predict_weights = spatial_weighting(
            self.cluster.cluster_centers_, Z, self.pairwise_distance
        )

        if is_regressor(self.estimator):
            piecewise_predict = np.array(
                list(map(lambda model: model.predict(X), self.list_estimators))
            )
            predicted_values = (overall_predict_weights.T * piecewise_predict.T).sum(1)
            return predicted_values
        else:
            raise NotImplementedError("Classification has not been implemented yet.")
