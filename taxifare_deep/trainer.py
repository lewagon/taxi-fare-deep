from taxifare_deep.data import get_data, clean_data
from taxifare_deep.utils import plot_model_history, simple_time_tracker
from taxifare_deep.preprocessor import create_pipeline
from taxifare_deep.network import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class Trainer:
    def __init__(self, nrows=10_000):
        self.df = get_data(nrows=nrows)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_train_preproc = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipe = None
        self.network = None
        self.history = None

    @simple_time_tracker
    def clean(self):
        print("###### loading and cleaning....")
        self.df = clean_data(self.df)
        self.X = self.df.drop("fare_amount", axis=1)
        self.y = self.df["fare_amount"]

    @simple_time_tracker
    def preproc(self, test_size=0.3):
        print("###### preprocessing....")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        self.pipe = create_pipeline()
        self.X_train_preproc = self.pipe.fit_transform(self.X_train)
        print(
            "###### shape of X_train_preproc, y_train: ",
            self.X_train_preproc.shape,
            self.y_train.shape,
        )

    @simple_time_tracker
    def fit(self, plot_history=True, verbose=1):
        print("###### fitting...")
        # TensorFlow cannot work with Sparse Matrix out of Sklearn's OHE
        self.X_train_preproc = self.X_train_preproc.todense()
        self.network = Network(input_dim=self.X_train_preproc.shape[1])
        print(self.network.model.summary())
        self.network.compile_model()
        self.history = self.network.fit_model(
            self.X_train_preproc, self.y_train, verbose=verbose
        )

        # Print & plot some key training results
        print("####### min val MAE", min(self.history.history["val_mae"]))
        print("####### epochs reached", len(self.history.epoch))
        if plot_history:
            plot_model_history(self.history)

    @simple_time_tracker
    def evaluate(self, X_test=None, y_test=None):
        """evaluates the pipeline on a test set and return the MAE"""
        # If no test set is given, use the holdout from train/test/split
        print("###### evaluates the model on a test set...")
        X_test = X_test or self.X_test
        y_test = y_test or self.y_test
        y_test_pred = self.network.model.predict(self.pipe.transform(X_test))

        print("###### test score (MAE)", mean_absolute_error(y_test, y_test_pred))
        # todo


if __name__ == "__main__":

    # Instanciate trainer with number of rows to download and use
    trainer = Trainer(nrows=5_000)

    # clean data
    trainer.clean()

    # Preprocess data and create train/test/split
    trainer.preproc(test_size=0.3)

    # Fit neural network and show training performance
    trainer.fit(plot_history=True, verbose=1)

    # evaluate on test set (by default the holdout from train/test/split)
    trainer.evaluate(X_test=None, y_test=None)
