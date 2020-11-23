from taxifare_deep.data import get_data, clean_data
from taxifare_deep.utils import compute_rmse, plot_model_history
from taxifare_deep.preprocessor import Preprocessor
from taxifare_deep.network import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class Trainer():
    def __init__(self, nrows=10_000):
        self.df = get_data(nrows=nrows)

    def clean(self):
        self.df = clean_data(self.df)
        self.X = self.df.drop("fare_amount", axis=1)
        self.y = self.df["fare_amount"]

    def preproc(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.preproc = Preprocessor().pipe.fit(self.X_train)
        self.X_train_preproc = self.preproc.transform(self.X_train)
        print('####### X_train_preproc shape', 'y_train shape #######')
        print(self.X_train_preproc.shape, self.y_train.shape)

    def fit(self, plot_history=True, verbose=1):
        self.network = Network(input_dim=self.X_train_preproc.shape[1])
        print(self.network.model.summary())
        self.network.compile_model()
        self.history = self.network.fit_model(self.X_train_preproc, self.y_train, verbose=verbose)

        # Print & plot some key training results
        print("####### min val MAE", min(self.history.history['val_mae']))
        print("####### epochs reached", len(self.history.epoch))
        if plot_history:
            plot_model_history(self.history)

    def evaluate(self, X_test=None, y_test=None):
        """evaluates the pipeline on a test set and return the MAE"""
        # If no test set is given, use the holdout from train/test/split
        X_test = X_test or self.X_test
        y_test = y_test or self.y_test

        y_test_pred = self.network.model.predict(self.preproc.transform(X_test))
        print('###### test score (MAE)', mean_absolute_error(y_test, y_test_pred))

if __name__ == "__main__":
    N = 10_000
    df = get_data(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    preproc = Preprocessor().pipe
    preproc.fit(X_train)
    X_train_preproc = preproc.transform(X_train)

    print('####### X_train_preproc shape', 'y_train shape #######')
    print(X_train_preproc.shape, y_train.shape)

    network = Network(input_dim=X_train_preproc.shape[1])
    print(network.model.summary())
    network.compile_model()
    history = network.fit_model(X_train_preproc, y_train)

    # Print & plot some key training results
    print("####### min val loss", min(history.history['val_mae']))
    print("####### epochs reached", len(history.epoch))
    plot_model_history(history)

    # Test error
    X_test_preproc = preproc.transform(X_test)
    y_test_pred = network.model.predict(X_test_preproc)
    print('###### test score (MAE)', mean_absolute_error(y_test, y_test_pred))
