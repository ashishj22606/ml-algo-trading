from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=20, batch_size=16, validation_data=None):
        """
        Train the LSTM model.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def predict(self, X):
        """
        Predict using the LSTM model.
        """
        return self.model.predict(X)
