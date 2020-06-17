from Core.ForecasterBase import RegressorBase
from tensorflow import keras
from tensorflow.keras.layers import Dense


class BasicRegressor(RegressorBase):
    """A basic implementation of an ANN based regressor.

    Since this is a ANN based regressor, the training data has to be
    univariate. The model consists of one fully connected layer with output
    size equal to forecast. This model is usually used to train one single
    ticker, and is not recommended for learning multiple stock data.

    Attributes
    ----------
    model: keras.models
        The keras model of the ANN
    """
    def buildModel(self, learningRate=None):
        if self.dataProcessor is None:
            message = ("DataProcessor not specified for this model. Either "
                       "load existing model or define a DataProcessor")
            raise Exception(message)

        model = keras.models.Sequential()
        model.add(Dense(self.forecast, input_shape=[self.lookBack]))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(loss="mean_squared_error", optimizer=optimizer,
                      metrics=['mse', "mae"])
        self.model = model


class DenseRegressor(RegressorBase):
    """A denser ANN implementation for stock price prediction

    Since this is a ANN based regressor, the training data has to be
    univariate. This model contains 7 fully connected layers with varying
    number of neurons with respect to forecast. This model can be used to learn
    from multiple tickers. The convertToWindowedDS method is written to handle
    data from multiple tickers.

    Attributes
    ----------
    model: keras.models
        The keras model of the ANN
    """
    def buildModel(self, learningRate=None):
        if self.dataProcessor is None:
            message = ("DataProcessor not specified for this model. Either "
                       "load existing model or define a DataProcessor")
            raise Exception(message)

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(self.lookBack,
                                     input_shape=[self.lookBack]))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*4))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(loss=keras.losses.Huber(), optimizer=optimizer,
                      metrics=['mse', "mae"])

        self.model = model
