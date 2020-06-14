from Core.DataProcessor import DataProcessor
import numpy as np
import math


class UniVarProcessor(DataProcessor):
    """A fully implemented DataProcessor for univariate data

    Parameters
    ----------
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    features: list
        A list of strings containing the names of the features to be included
    tickers: list
        A list of tickers for which the model is to be trained/inferenced
    isSeq2Seq: bool
        A boolean indicating whether the data needs to be prepared as sequence
        to sequence or not
    tickerData: dict
        A dictionary containing all the ticker data
    """
    def __init__(self, tickers, features, lookBack, forecast, isSeq2Seq=False):
        super().__init__(tickers, features)

        self.lookBack = lookBack
        self.forecast = forecast
        self.tickers = tickers
        self.features = features
        self.isSeq2Seq = isSeq2Seq

        self.tickerData = self.getTickerData()

    def inputProcessor(self, data, context):
        ticker = context["ticker"]
        close = data["Close"]
        if context["isTrain"]:
            X, Y = self.convertToWindows(close, True)
            return np.array(X), np.array(Y)
        else:
            X = self.convertToWindows(close, False)
            return np.array(X)

    def outputProcessor(self, modelOut, context):
        return modelOut

    def getTrainingData(self):
        context = {}
        context["isTrain"] = True

        allX = None
        allY = None
        for i, ticker in enumerate(self.tickers):
            context["ticker"] = ticker
            X, Y = self.inputProcessor(self.tickerData[ticker], context)
            if i == 0:
                allX = X
                allY = Y
            else:
                allX = np.concatenate((allX, X), axis=0)
                allY = np.concatenate((allY, Y), axis=0)
        if self.isSeq2Seq:
            return (allX.reshape(-1, self.lookBack, 1),
                    allY.reshape(-1, self.lookBack, 1))
        else:
            return allX, allY

    def convertToWindows(self, data, isTrain, shuffle=False):
        """Converts the input data to a windowed dataset

        Parameters
        ----------
        data: np.array
            The specific ticker data
        isTrain: bool
            A boolean indicating whether to create a target
        shuffle: bool, optional
            A boolean indicating whether the data should be shuffled
        """
        data = data.values
        if isTrain:
            windowSize = self.lookBack + self.forecast
            windowData = self.splitArray(data, windowSize, dropRemaining=True)
            if shuffle:
                np.random.shuffle(windowData)

            if not self.isSeq2Seq:
                X = [item[:-self.forecast] for item in windowData]
                Y = [item[-self.forecast:] for item in windowData]
            else:
                X = [item[:-self.forecast] for item in windowData]
                Y = [item[self.forecast:] for item in windowData]

            return X, Y
        else:
            windowData = self.splitArray(data, self.lookBack,
                                         dropRemaining=True)
            return windowData

    def splitArray(self, array, unitLength, dropRemaining=False):
        """A method to split an array into equal sized chunks

        Parameters
        ----------
        array: list
            The array that is to be split
        unitLength: int
            The chunk size
        dropRemaining: bool, optional
            Indicates whether leftover elements should be kept
        """
        start = 0
        finalAr = []
        while True:
            end = start + unitLength
            if end > len(array):
                if not dropRemaining:
                    finalAr.append(array[end-unitLength:])
                break
            finalAr.append(array[start:end])
            start = end
        return finalAr


class MultiVarProcessor(DataProcessor):
    """A fully implemented DataProcessor for multivariate data

    Parameters
    ----------
    lookBack: int, optional
        Variable to specify how many days to consider when making
        a prediction
    forecast: int, optional
        Variable to specify how many days ahead to make predictions for.
    yInd: int
        The column position of the target
    features: list
        A list of strings containing the names of the features to be included
    tickers: list
        A list of tickers for which the model is to be trained/inferenced
    isSeq2Seq: bool
        A boolean indicating whether the data needs to be prepared as sequence
        to sequence or not
    tickerData: dict
        A dictionary containing all the ticker data
    """
    def __init__(self, tickers, features, lookBack, forecast,
                 targetFeature, isSeq2Seq=False):
        super().__init__(tickers, features)

        self.lookBack = lookBack
        self.forecast = forecast
        self.yInd = self.allFeatures.index(targetFeature)
        self.tickers = tickers
        self.features = features
        self.isSeq2Seq = isSeq2Seq

        self.tickerData = self.getTickerData()

    def inputProcessor(self, data, context):
        ticker = context["ticker"]
        tickerData = data
        if context["isTrain"]:
            X, Y = self.convertToWindows(tickerData, True)
            return np.array(X), np.array(Y)
        else:
            X = self.convertToWindows(tickerData, False)
            return np.array(X)

    def outputProcessor(self, modelOut, context):
        return modelOut

    def getTrainingData(self):
        context = {}
        context["isTrain"] = True

        allX = None
        allY = None
        splitRatio = 0.8
        for i, ticker in enumerate(self.tickers):
            context["ticker"] = ticker
            X, Y = self.inputProcessor(self.tickerData[ticker], context)
            if i == 0:
                allX = X
                allY = Y
            else:
                allX = np.concatenate((allX, X), axis=0)
                allY = np.concatenate((allY, Y), axis=0)
        if self.isSeq2Seq:
            splitIndex = math.floor(splitRatio * len(allX))
            return allX[:splitIndex], allY[:splitIndex], allX[splitIndex:], allY[splitIndex:]
        else:
            splitIndex = math.floor(splitRatio * len(allX))
            return allX[:splitIndex], allY[:splitIndex], allX[splitIndex:], allY[splitIndex:]

    def convertToWindows(self, data, isTrain, shuffle=False):
        """Converts the input data to a windowed dataset

        Parameters
        ----------
        data: np.array
            The specific ticker data
        isTrain: bool
            A boolean indicating whether to create a target
        shuffle: bool, optional
            A boolean indicating whether the data should be shuffled
        """
        data = data.values
        if isTrain:
            windowSize = self.lookBack + self.forecast
            windowData = self.splitArray(data, windowSize, dropRemaining=True)
            if shuffle:
                np.random.shuffle(windowData)

            if not self.isSeq2Seq:
                X = [item[:-self.forecast] for item in windowData]
                Y = [item[-self.forecast:, self.yInd] for item in windowData]
            else:
                X = [item[:-self.forecast] for item in windowData]
                Y = [item[self.forecast:, self.yInd] for item in windowData]

            return X, Y
        else:
            windowData = self.splitArray(data, self.lookBack,
                                         dropRemaining=True)
            return windowData

    def splitArray(self, array, unitLength, dropRemaining=False):
        """A method to split an array into equal sized chunks

        Parameters
        ----------
        array: list
            The array that is to be split
        unitLength: int
            The chunk size
        dropRemaining: bool, optional
            Indicates whether leftover elements should be kept
        """
        start = 0
        finalAr = []
        while True:
            end = start + unitLength
            if end > len(array):
                if not dropRemaining:
                    finalAr.append(array[end-unitLength:])
                break
            finalAr.append(array[start:end])
            start = end
        return finalAr
