"""Tests for `gaia.ml` package."""


import numpy as np
import pytest
from sklearn.cluster import KMeans

from gaia.ml import SpatialModel, WrongEstimator


@pytest.fixture
def spatial_model_class():
    """Initialize a SpatialModel class."""
    spatial_model = SpatialModel()
    return spatial_model

def test_spatial_model_raises_exception_machine_bad_estimator():
    with pytest.raises(WrongEstimator):
        SpatialModel(estimator=KMeans)

def test_spatial_model_fit(spatial_model_class):
    X = np.random.uniform(-10, 10, (1000, 5))
    Z = np.random.uniform(-10, 10, (1000, 2))
    y = (X**2).sum(1) + np.sin(Z).sum(1)

    spatial_model_class.fit(X, Z, y)
    assert len(spatial_model_class.list_estimators)==spatial_model_class.cluster.n_clusters

def test_spatial_model_fit_predict(spatial_model_class):
    X = np.random.uniform(-10, 10, (1000, 5))
    Z = np.random.uniform(-10, 10, (1000, 2))
    y = (X**2).sum(1) + np.sin(Z).sum(1)

    X_pred = np.random.uniform(-10, 10, (1000, 5))
    Z_pred = np.random.uniform(-10, 10, (1000, 2))
    y_pred = (X**2).sum(1) + np.sin(Z).sum(1)
    
    spatial_model_class.fit(X, Z, y)
    assert spatial_model_class.predict(X_pred, Z_pred).shape==y_pred.shape
