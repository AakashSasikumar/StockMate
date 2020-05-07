from Models.KerasBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math


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
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    model: keras.models
        The keras model of the ANN
    yInd: int
        The position of the target variable
    """
    def __init__(self, numDims, lookBack=4, forecast=1, loadLatest=False):
        self.numDims = numDims
        self.lookBack = lookBack
        self.forecast = forecast
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
        model.add(LSTM(200, input_shape=(self.lookBack, self.numDims),
                       stateful=False, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))

        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()

        model.compile(optimizer=optimizer, loss=keras.losses.Huber(),
                      metrics=['mse'])

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

    def convertToWindowedDS(self, data, yInd=0, splitRatio=0.7, batchSize=32,
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
            if self.numDims == 1:
                values = data[ticker].values.reshape(-1, 1)
            else:
                values = data[ticker].values
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
