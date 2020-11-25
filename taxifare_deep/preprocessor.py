from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from taxifare_deep.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash

from sklearn.preprocessing import FunctionTransformer

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

    time_pipe = make_pipeline(
        TimeFeaturesEncoder("pickup_datetime"),
        OneHotEncoder(handle_unknown="ignore", sparse=True)
    )

    final_pipe = ColumnTransformer(
        [
            ("time_preproc", time_pipe, ["pickup_datetime"]),
            ("geohash", geohash_pipe, lonlat_features),
            ("dist_preproc", dist_pipe, lonlat_features),
            ("passenger_scaler", StandardScaler(), ["passenger_count"]),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    return final_pipe
