"""Main module."""

import copy
import numpy as np

from sklearn.base import is_classifier, is_regressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances

from gaia.utils import spatial_weighting
from gaia.kernel import Kernel


class SpatialModel:
    """Base class for spatial modeling."""

    def __init__(
        self, estimator=LinearRegression(), n_clusters=8, cluster_random_state=None
    ):
        """Initialization of SpatialModel.

        Parameters
        ----------
        n_clusters: int
            Number of clusters to build. Default `sklearn.cluster.KMeans` value.
        estimator: sklearn.base.BaseEstimator
            A sklearn model with fit / predict paradigm.
        pairwise_distance: sklearn.metrics.pairwise
            A distance metric
        """
        # declare machine learning method
        if is_classifier(estimator) or is_regressor(estimator):
            self.estimator = estimator
        else:
            raise TypeError(
                "It does not seem to be a scikit-learn estimator "
                "for supervised learning."
            )

        # declare clustering method
        self.cluster = KMeans(n_clusters=n_clusters, random_state=cluster_random_state)

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
        # fit clustering
        self.cluster.fit(Z)

        # iterate over clusters (a model by cluster)
        self.list_estimators = list()
        for num_model in range(self.cluster.n_clusters):
            model_mask = self.cluster.labels_ == num_model
            current_model = copy.deepcopy(self.estimator)
            current_model.fit(X[model_mask], y[model_mask])
            self.list_estimators.append(current_model)

    def predict(self, X, Z, pairwise_dist=euclidean_distances, kernel_norm=Kernel()):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Z : {array-like, sparse matrix} of shape (n_samples, n_coordinates)
            Training spatial coordinates, where `n_samples` is the number of samples and
            `n_coordinates` is the number of coordinates, for instance **lat** and **lng**.
        pairwise_dist: sklearn.metrics.pairwise
            Spatial autocorrelation a priori.
        kernel: gaia.kernel.Kernel class
            Kernel used to moderate cluster distance.

        Returns
        -------
        predicted_values: array, shape (n_samples,)
            Returns predicted values.
        """
        if is_regressor(self.estimator):
            # compute clustering weights
            weighted_cluster_dist = spatial_weighting(
                self.cluster.cluster_centers_, Z, pairwise_dist
            )
            weighted_cluster_dist = (
                weighted_cluster_dist.T - weighted_cluster_dist.min(1)
            ).T

            # compute final cluster weights and predictions
            piecewise_weights = kernel_norm.kernel(weighted_cluster_dist)
            piecewise_predict = np.array(
                list(map(lambda model: model.predict(X), self.list_estimators))
            )
            predicted_values = (piecewise_weights * piecewise_predict.T).sum(1)
            return predicted_values
        else:
            raise NotImplementedError("Classification has not been implemented yet.")
