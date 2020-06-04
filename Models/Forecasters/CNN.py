from Core.ForecasterBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Conv2D
import math


class BasicCNN(RegressorBase):
    """A basic implementation of a CNN based regressor for stock price prediction.

    This model works with multivariate data. The model structure has a
    convolution layer, two LSTM layers with 32 units and a dense layer.

    Attributes
    ----------
    numDims: int
        The dimensionality of the data
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    yInd: int
        The column index of the target
    model: keras.models
        The keras model
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
        self.numDims = len(self.dataProcessor.features)
        model = keras.models.Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, strides=1,
                         input_shape=[self.lookBack, self.numDims],
                         padding="causal", activation='relu'))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.Dense(1))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()
        model.compile(loss="mse", optimizer=optimizer,
                      metrics=["mse"])

        self.model = model


class WaveNet(RegressorBase):
    """A keras implementation of wavenet for stock prediction.

    This model works with multivariate data.

    Attributes
    ----------
    numDims: int
        The dimensionality of the data
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    yInd: int
        The column index of the target
    model: keras.models
        The keras model
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
        self.numDims = len(self.dataProcessor.features)
        model = keras.models.Sequential()
        for dilation_rate in (1, 2, 4, 8, 16, 32):
            model.add(
                keras.layers.Conv1D(filters=32,
                                    kernel_size=2,
                                    strides=1,
                                    dilation_rate=dilation_rate,
                                    padding="causal",
                                    activation="relu")
                )
        model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()
        model.compile(loss=keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"])

        self.model = model
