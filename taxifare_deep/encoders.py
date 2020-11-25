import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from taxifare_deep.utils import haversine_vectorized, sinuser, cosinuser
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
        print("###### start TimeFeaturesEncoder...")
        start_time = time.time()
        assert isinstance(X, pd.DataFrame)
        pickup_dt = pd.to_datetime(
            X[self.time_column], format="%Y-%m-%d %H:%M:%S UTC", utc=True
        )
        pickup_dt = pickup_dt.dt.tz_convert(self.time_zone_name).dt
        dow = pickup_dt.weekday
        hour = pickup_dt.hour
        hour_sin = sinuser(hour, 24)
        hour_cos = cosinuser(hour, 24)
        month = pickup_dt.month
        month_sin = sinuser(month, 12)
        month_cos = cosinuser(month, 12)
        year = pickup_dt.year
        print("###### TimeFeaturesEncoder time elasped (s):", time.time() - start_time)
        return pd.concat([dow, year, hour_sin, hour_cos, month_sin, month_cos], axis=1)


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
        print("##### start DistanceTransformer...")
        start_time = time.time()
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(
            X_,
            start_lat=self.start_lat,
            start_lon=self.start_lon,
            end_lat=self.end_lat,
            end_lon=self.end_lon,
        )
        print("###### DistanceTransformer time elasped (s):", time.time() - start_time)
        return X_[["distance"]]
