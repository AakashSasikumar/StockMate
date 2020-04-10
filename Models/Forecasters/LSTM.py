from Models.KerasBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math


class BasicLSTM(RegressorBase):
    """A basic implementation of an LSTM based NN for stock price prediction.

    Since this is an LSTM based regressor, the dimensionality of the data can
    be greater than or equal to 1.

    This model contains 3 LSTM layers with size 50 along with dropouts, and
    three Dense layers with varying size as per forecast size.

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
        model.add(LSTM(units=50, return_sequences=True,
                       input_shape=(self.lookBack, 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50))
        model.add(Dropout(0.3))
        model.add(Dense(self.forecast*3))
        model.add(Dense(self.forecast*2))
        model.add(Dense(self.forecast))

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
        windows that can be later concatenated with other ticker data.

        Parameters
        ----------
        data: pd.Series()
            The ticker data
        yInd:
            The index of the variable that is to be predicted
        """
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast,
                       shift=self.forecast + self.lookBack,
                       drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.map(lambda w: (w[:-self.forecast], w[-self.forecast:, yInd]))
        return ds

    def convertToWindowedDS(self, data, yInd, splitRatio=0.7, batchSize=32, shuffle=False):
        """Method to convert the given dataset into windowed form for training.

        Parameters
        ----------
        data: pd.DataFrame()
            The input data. This can have multiple columns for multiple
            tickers.
        yInd: int
            The position of the variable to be predicted
        splitRatio: float, optional
            The train, validation split ratio for the input data
        batchSize: int, optional
            The batchsize for the resultant windowed dataset
        shuffle: bool, optional
            If true, the windowed data will be shuffled

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
