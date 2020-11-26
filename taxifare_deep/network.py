from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


class Network:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        reg = regularizers.l1_l2(l1=0.005, l2=0.0005)
        model = Sequential()
        model.add(
            layers.Dense(
                200, input_dim=self.input_dim, activation="relu", kernel_regularizer=reg
            )
        )
        layers.Dropout(rate=0.2)
        model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
        layers.Dropout(rate=0.2)
        model.add(layers.Dense(20, activation="relu", kernel_regularizer=reg))
        layers.Dropout(rate=0.2)
        model.add(layers.Dense(1, activation="linear"))
        self.model = model

    def compile_model(self):
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])
        return self.model

    def fit_model(self, X, y, verbose=1):
        es = EarlyStopping(
            monitor="val_loss", patience=5, verbose=0, restore_best_weights=True
        )

        history = self.model.fit(
            X,
            y,
            validation_split=0.3,
            epochs=500,
            batch_size=64,
            callbacks=[es],
            verbose=verbose,
            workers=-1,
            use_multiprocessing=True,
        )
        return history
