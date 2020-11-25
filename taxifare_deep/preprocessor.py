from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from taxifare_deep.encoders import TimeFeaturesEncoder, DistanceTransformer

def create_pipeline():
    dist_features = [
            "pickup_latitude",
            "pickup_longitude",
            'dropoff_latitude',
            'dropoff_longitude']

    dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])

    col_trans1 = ColumnTransformer([
        ('time_proc', TimeFeaturesEncoder('pickup_datetime'), ['pickup_datetime']),
        ('dist_proc', dist_pipe, dist_features),
        ('passenger_proc', StandardScaler(), ['passenger_count'])],
        remainder="drop",
        n_jobs=-1)

    col_trans2 = ColumnTransformer([
        # one hot encode 'dow' and 'year' (the first two columns out of col_trans1)
        ('ohe', OneHotEncoder(handle_unknown='ignore'), [0, 1])],
        remainder="passthrough",
        n_jobs=-1)

    return make_pipeline(col_trans1, col_trans2)
