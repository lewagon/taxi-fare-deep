import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def haversine_vectorized(
    df,
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version of the haversine distance for pandas df
    Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(
        df[start_lon].astype(float)
    )
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(
        df[end_lon].astype(float)
    )
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def sinuser(X, period):
    return np.sin(2 * math.pi / period * X)


def cosinuser(X, period):
    return np.cos(2 * math.pi / period * X)


def plot_model_history(history):
    """Plot a Keras-fitted model history"""

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Mean Square Error - Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(history.history["mae"])
    plt.plot(history.history["val_mae"])
    plt.title("Model mae")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="best")
    plt.show()


def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return:
    """
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed

