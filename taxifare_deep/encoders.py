import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from taxifare_deep.utils import haversine_vectorized, sinuser, cosinuser
import pygeohash as gh
import time


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    Extract the day of week (dow), the hour, the month and the year from a time column.
    """

    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start_time = time.time()
        assert isinstance(X, pd.DataFrame)
        pickup_dt = pd.to_datetime(
            X[self.time_column], format="%Y-%m-%d %H:%M:%S UTC", utc=True
        )
        pickup_dt = pickup_dt.dt.tz_convert(self.time_zone_name).dt
        dow = pickup_dt.weekday
        hour = pickup_dt.hour
        month = pickup_dt.month
        year = pickup_dt.year
        #hour_sin = sinuser(hour, 24)
        #hour_cos = cosinuser(hour, 24)
        #month_sin = sinuser(month, 12)
        #month_cos = cosinuser(month, 12)
        return pd.concat([hour, dow, month, year], axis=1)


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Compute the haversine distance between two GPS points.
    Returns a copy of the DataFrame X with only one column: 'distance'
    """

    def __init__(
        self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude",
    ):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon,
        )
        return X_[["distance"]]

class AddGeohash(BaseEstimator, TransformerMixin):
    '''
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon,lat) tuple, for pick-up, and drop-off
    '''

    def __init__(self, precision=5):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(
            lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=self.precision), axis=1)
        X['geohash_dropoff'] = X.apply(
            lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=self.precision), axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]
