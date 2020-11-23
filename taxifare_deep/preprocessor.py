from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from taxifare_deep.encoders import TimeFeaturesEncoder, DistanceTransformer

class Preprocessor():

    def __init__(self):
        '''initialize the pipeline'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('stdscaler', StandardScaler())
        ])
        pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                "pickup_latitude",
                "pickup_longitude",
                'dropoff_latitude',
                'dropoff_longitude'
            ]),
            ('time', time_pipe, ['pickup_datetime'])],
            remainder="drop",  # Drop all other features to start with
            n_jobs=-1)
        self.pipe = pipe
