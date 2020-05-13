from Models.KerasBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Conv2D
import math


class BasicCNN(RegressorBase):
    """A basic implementation of a CNN based regressor for stock price prediction.

    Theoretically the CNN model can support multivariate data. However, that
    support hasn't been built into this yet. The model structure has a
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
    model: keras.models
        The keras model
    """
    def __init__(self, numDims, lookBack=4, forecast=1, loadLatest=False):
        self.numDims = numDims
        self.lookBack = lookBack
        self.forecast = forecast
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=None):
        model = keras.models.Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, strides=1,
                         padding="causal", activation='relu'))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.Dense(1))
        if learningRate:
            optimizer = keras.optimizers.SGD(lr=learningRate, momentum=0.9)
        else:
            optimizer = keras.optimizers.SGD(momentum=0.9)
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer,
                      metrics=["mae"])

        self.model = model

    def convertToWindows(self, data, yInd):
        """Method to convert the given ticker data into windows.

        This method deals with individual ticker data and converts them to
        windows that can be later concatenated with other ticker data. This is
        modified to return a sequence that is moved forward by forecast.

        Parameters
        ----------
        data: pd.Series()
            The ticker data
        yInd:
            The index of the variable that is to be predicted
        """
        self.yInd = yInd
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast,
                       shift=self.forecast,
                       drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.map(lambda w: (w[:-self.forecast], w[self.forecast:, yInd]))
        return ds

    def convertToWindowedDS(self, data, yInd=0, splitRatio=0.7, batchSize=64,
                            shuffle=True, columns=None):
        """Method to convert the given dataset into windowed form for training.

        Parameters
        ----------
        data: pd.DataFrame()
            The input data. This can have multiple columns for multiple
            tickers.
        yInd: int, optional
            The position of the variable to be predicted
        splitRatio: float, optional
            The train, validation split ratio for the input data
        batchSize: int, optional
            The batchsize for the resultant windowed dataset
        shuffle: bool, optional
            If true, the windowed data will be shuffled
        columns: list, optional
            If columns are specified, the windowed dataset is
            made from the data from these columns

        Returns
        -------
        trainDS: tf.Dataset
            The windowed form of the respective amount of data for training
        validDS: tf.Dataset
            The windowed form of the respective amount of data for validation
        """
        lenTrain = 0
        lenValid = 0
        if isinstance(columns, list):
            cols = columns
        else:
            cols = list(columns)
        for i, ticker in enumerate(cols):
            values = data[ticker].values.reshape(-1, 1)
            splitInd = math.floor(splitRatio * len(values))
            train = values[:splitInd]
            valid = values[splitInd:]
            lenTrain += len(train)
            lenValid += len(valid)

            if i == 0:
                trainDS = self.convertToWindows(train, yInd)
                validDS = self.convertToWindows(valid, yInd)
            else:
                tmpTrain = self.convertToWindows(train, yInd)
                tmpValid = self.convertToWindows(valid, yInd)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        if shuffle:
            trainDS = trainDS.shuffle(lenTrain)
            validDS = validDS.shuffle(lenValid)
        trainDS = trainDS.batch(batchSize).prefetch(1)
        validDS = validDS.batch(batchSize).prefetch(1)

        return trainDS, validDS
