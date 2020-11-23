from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping

class Network():

    def __init__(self, input_dim):
        self.input_dim = input_dim
        model = Sequential()
        model.add(layers.Dense(100, input_dim=self.input_dim, activation='relu'))
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        self.model = model

    def compile_model(self):
        self.model.compile(
            loss="mean_squared_error", optimizer='adam', metrics=['mae'])
        return self.model

    def fit_model(self, X, y, verbose=1):
        es = EarlyStopping(
            monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)

        history = self.model.fit(X, y, validation_split=0.3,
                  epochs=400, batch_size=64, callbacks=[es],
                  verbose=verbose, workers=-1, use_multiprocessing=True)
        return history
