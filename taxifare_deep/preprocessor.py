from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from taxifare_deep.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash

from sklearn.preprocessing import FunctionTransformer
import ipdb

def create_pipeline():
    lonlat_features = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    dist_pipe = make_pipeline(
        DistanceTransformer(),
        StandardScaler()
    )

    geohash_pipe = make_pipeline(
        AddGeohash(precision=5),
        OneHotEncoder(handle_unknown="ignore", sparse=True)
    )

    col_trans1 = ColumnTransformer(
        [
            ("time_preproc", TimeFeaturesEncoder("pickup_datetime"), ["pickup_datetime"]),
            ("geohash", geohash_pipe, lonlat_features),
            ("dist_preproc", dist_pipe, lonlat_features),
            ("passenger_scaler", StandardScaler(), ["passenger_count"]),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    # One-hot-encode 'dow' and 'year' columns created from col_trans1
    # at index [0,1]
    col_trans2 = ColumnTransformer(
        [("ohe_dates", OneHotEncoder(handle_unknown="ignore"), [0, 1])],
        remainder="passthrough",
        n_jobs=-1,
    )

    return make_pipeline(col_trans1)#, col_trans2)
