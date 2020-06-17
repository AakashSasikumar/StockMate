from Core.ForecasterBase import RegressorBase
from tensorflow import keras


class BasicCNN(RegressorBase):
    """A basic implementation of a CNN based regressor for stock price prediction.

    This model works with multivariate data. The model structure has a
    convolution layer, two LSTM layers with 32 units and a dense layer.

    Attributes
    ----------
    numDims: int
        The dimensionality of the data
    model: keras.models
        The keras model
    """
    def buildModel(self, learningRate=None):
        if self.dataProcessor is None:
            message = ("DataProcessor not specified for this model. Either "
                       "load existing model or define a DataProcessor")
            raise Exception(message)
        self.numDims = len(self.dataProcessor.features)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
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
                      metrics=["mse", "mae"])

        self.model = model


class WaveNet(RegressorBase):
    """A keras implementation of wavenet for stock prediction.

    The implementation of the revered WaveNet. This model supports
    multivariate data as well as univariate data.

    Attributes
    ----------
    numDims: int
        The dimensionality of the data
    model: keras.models
        The keras model
    """
    def buildModel(self, learningRate=None):
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
                      metrics=["mae", "mse"])

        self.model = model
