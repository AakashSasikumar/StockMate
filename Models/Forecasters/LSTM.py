from Core.ForecasterBase import RegressorBase
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout


class BasicLSTM(RegressorBase):
    """A basic implementation of an LSTM for stock price prediction.

    Since this is an LSTM based regressor, the dimensionality of the data can
    be greater than or equal to 1. This model can be used to train on an
    entire index. This model contains 2 LSTM layers with size 200 and 150,
    along with dropouts, and a Dense layer for the return sequence.

    Attributes
    ----------
    numDims: int
        The dimensionality of the input data.
    model: keras.models
        The keras model of the ANN
    """
    def buildModel(self, learningRate=None):
        if self.dataProcessor is None:
            message = ("DataProcessor not specified for this model. Either "
                       "load existing model or define a DataProcessor")
            raise Exception(message)
        self.numDims = len(self.dataProcessor.features)
        model = keras.models.Sequential()
        model.add(LSTM(200, input_shape=(self.lookBack, self.numDims),
                       stateful=False, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='relu'))

        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(optimizer=optimizer, loss=keras.losses.Huber(),
                      metrics=['mse', 'mae'])

        self.model = model


class DenseLSTM(RegressorBase):
    """A basic implementation of an LSTM for stock price prediction.

    Since this is an LSTM based regressor, the dimensionality of the data can
    be greater than or equal to 1. This model can be used to train on an
    entire index. This model contains 3 LSTM layers with size 200, 200 and 150,
    along with dropouts, and a Dense layer for the return sequence.

    Attributes
    ----------
    numDims: int
        The dimensionality of the input data.
    model: keras.models
        The keras model of the ANN
    """
    def buildModel(self, learningRate=None):
        if self.dataProcessor is None:
            message = ("DataProcessor not specified for this model. Either "
                       "load existing model or define a DataProcessor")
            raise Exception(message)
        self.numDims = len(self.dataProcessor.features)

        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(200,
                                    input_shape=(self.lookBack, self.numDims),
                                    return_sequences=True))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.LSTM(200, return_sequences=True))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.LSTM(150, return_sequences=True))
        model.add(keras.layers.Dense(1, activation='relu'))

        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(loss='mse', optimizer=optimizer, metrics=["mse"])

        self.model = model
