from Models.KerasBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import math


class BasicRegressor(RegressorBase):
    """A basic implementation of an ANN based regressor.

    Since this is a ANN based regressor, the training data has to be
    univariate.

    The model consists of one fully connected layer with output size
     equal to forecast. This model is usually used to train one single ticker,
     and is not recommended for learning multiple stock data.

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
    def __init__(self, lookBack=4, forecast=1, loadLatest=False):
        self.lookBack = lookBack
        self.forecast = forecast
        self.model = None
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=None):
        """Builds the model and sets the class attribute

        Parameters
        ----------
        learningRate: float, optional
            Optional learning to specify for the AdamOptimizer

        """
        model = keras.models.Sequential()
        model.add(Dense(self.forecast, input_shape=[self.lookBack]))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(loss="mean_squared_error", optimizer=optimizer,
                      metrics=['mse'])
        self.model = model

    def convertToWindowedDS(self, data, batchSize=32):
        """Converts series data into windows for training

        Parameters
        ----------
        data: pd.Series
            The data to be converted into windowed form
        batchSize: int, optional
            The batchsize for the resulting windowed dataset

        Returns
        -------
        ds: tf.Dataset
            The windowed dataset
        """
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast,
                       shift=self.forecast, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.shuffle(len(data))
        ds = ds.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        ds = ds.batch(batchSize).prefetch(1)
        return ds


class DenseRegressor(RegressorBase):
    """A denser ANN implementation for stock price prediction

    Since this is a ANN based regressor, the training data has to be
    univariate.

    This model contains 7 fully connected layers with varying number
    of neurons with respect to forecast. This model can be used to learn
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
    def __init__(self, lookBack=4, forecast=1, loadLatest=False):
        self.lookBack = lookBack
        self.forecast = forecast
        self.model = None
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=None):
        """Builds the model and sets the class attribute

        Parameters
        ----------
        learningRate: float, optional
            Optional learning to specify for the AdamOptimizer

        """
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
                      metrics=['mae'])

        self.model = model

    def convertToWindows(self, data):
        """Method to convert the given ticker data into windows.

        This method deals with individual ticker data and converts them to
        windows that can be later concatenated with other ticker data.

        Parameters
        ----------
        data: pd.Series()
            The ticker data
        """
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast, shift=self.forecast,
                       drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        return ds

    def convertToWindowedDS(self, data, splitRatio=0.7, batchSize=32):
        """Method to convert the given dataset into windowed form for training.

        Parameters
        ----------
        data: pd.DataFrame()
            The input data. This can have multiple columns for multiple
            tickers.
        splitRatio: float, optional
            The train, validation split ratio for the input data
        batchSize: int, optional
            The batchsize for the resultant windowed dataset

        Returns
        -------
        trainDS: tf.Dataset
            The windowed form of the respective amount of data for training
        validDS: tf.Dataset
            The windowed form of the respective amount of data for validation
        """
        lenTrain = 0
        lenValid = 0
        for i, ticker in enumerate(list(data.columns)):
            values = data[ticker].values
            splitInd = math.floor(splitRatio * len(values))
            train = values[:splitInd]
            valid = values[splitInd:]
            lenTrain += len(train)
            lenValid += len(valid)

            if i == 0:
                trainDS = self.convertToWindows(train)
                validDS = self.convertToWindows(valid)
            else:
                tmpTrain = self.convertToWindows(train)
                tmpValid = self.convertToWindows(valid)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        trainDS = trainDS.shuffle(lenTrain)
        validDS = validDS.shuffle(lenValid)
        trainDS = trainDS.batch(batchSize).prefetch(1)
        validDS = validDS.batch(batchSize).prefetch(1)

        return trainDS, validDS
