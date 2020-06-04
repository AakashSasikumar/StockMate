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
    dataProcessor: Core.DataProcessor
        The implementation of data processor for this model
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    model: keras.models
        The keras model of the ANN
    """
    def __init__(self):
        self.dataProcessor = None

    def buildModel(self, learningRate=None):
        """Builds the model and sets the class attribute

        Parameters
        ----------
        learningRate: float, optional
            Optional learning to specify for the AdamOptimizer

        """
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
                      metrics=['mse'])
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
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    model: keras.models
        The keras model of the ANN
    """
    def __init__(self):
        self.dataProcessor = None

    def buildModel(self, learningRate=None):
        """Builds the model and sets the class attribute

        Parameters
        ----------
        learningRate: float, optional
            Optional learning to specify for the AdamOptimizer

        """
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
        # optimizer = keras.optimizers.RMSprop(lr=learningRate)
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

    #     model.compile(loss="mean_squared_error", optimizer=optimizer,
    #                   metrics=['mse'])
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer,
                      metrics=['mse'])

        self.model = model
