from Core.DataProcessor import DataProcessor
import numpy as np
from tensorflow import data as DS
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
                 targetFeature, isSeq2Seq=False, shuffle=True,
                 validationSplit=0.8, batchSize=32):
        super().__init__(tickers, features)

        self.lookBack = lookBack
        self.forecast = forecast
        self.yInd = self.allFeatures.index(targetFeature.lower())
        self.tickers = tickers
        self.features = features
        self.isSeq2Seq = isSeq2Seq

        self.tickerData = self.getTickerData()
        self.shuffle = shuffle
        self.batchSize = batchSize
        self.validationSplit = validationSplit

    def inputProcessor(self, data, context):
        tickerData = data
        if context["isTrain"]:
            ds = self.convertToWindows(tickerData, True)
            return ds
        else:
            ds = self.convertToWindows(tickerData, False)
            return ds

    def outputProcessor(self, modelOut, context):
        return modelOut

    def getTrainingData(self, shuffle=True):
        context = {}
        context["isTrain"] = True
        lenTrain = 0
        lenValid = 0
        trainDS = None
        validDS = None
        for i, ticker in enumerate(self.tickers):
            context["ticker"] = ticker
            data = self.tickerData[ticker]
            splitIndex = math.floor(self.validationSplit * len(data))
            lenTrain += len(data[:splitIndex])
            lenValid += len(data[splitIndex:])
            if i == 0:
                trainDS = self.inputProcessor(data[:splitIndex], context)
                validDS = self.inputProcessor(data[splitIndex:], context)
            else:
                tmpTrain = self.inputProcessor(data[:splitIndex], context)
                tmpValid = self.inputProcessor(data[splitIndex:], context)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        if shuffle:
            trainDS.shuffle(lenTrain)
            validDS.shuffle(lenValid)
        trainDS = trainDS.batch(self.batchSize).prefetch(1)
        validDS = validDS.batch(self.batchSize).prefetch(1)
        return trainDS, validDS

    def convertToWindows(self, data, isTrain):
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
        dataset = DS.Dataset.from_tensor_slices(data)
        if isTrain:
            dataset = dataset.window(self.lookBack + self.forecast,
                                     shift=self.forecast,
                                     drop_remainder=True)
            dataset = dataset.flat_map(
                lambda w: w.batch(self.lookBack + self.forecast))
            if self.isSeq2Seq:
                dataset = dataset.map(lambda w: (
                    w[:-self.forecast], w[self.forecast:, self.yInd]))
            else:
                dataset = dataset.map(lambda w: (
                    w[:-self.forecast], w[-self.forecast:, self.yInd]))
            return dataset
        else:
            dataset = dataset.window(self.lookBack, shift=self.forecast,
                                     drop_remainder=True)
            dataset = dataset.flat_map(lambda w: w.batch(self.lookBack))
            dataset = dataset.batch(1).prefetch(1)
            return dataset


class testProcessor(DataProcessor):
    def __init__(self, tickers, features, lookBack, forecast,
                 targetFeature, isSeq2Seq=False, shuffle=True,
                 validationSplit=0.9, batchSize=128):
        super().__init__(tickers, features)

        self.lookBack = lookBack
        self.forecast = forecast
        self.yInd = self.allFeatures.index(targetFeature.lower())
        self.targetFeature = targetFeature
        self.tickers = tickers
        self.features = features
        self.isSeq2Seq = isSeq2Seq

        self.tickerData = self.getTickerData()
        self.shuffle = shuffle
        self.batchSize = batchSize
        self.validationSplit = validationSplit

    def inputProcessor(self, data, context):
        tickerData = self.tickerData[context['ticker']]
        data = data.copy()
        for feature in self.features:
            data[feature] = (data[feature] - min(tickerData[feature])) /\
                            (max(tickerData[feature]) -
                                min(tickerData[feature]))
        if context["isTrain"]:
            ds = self.convertToWindows(data, True)
            return ds
        else:
            ds = self.convertToWindows(data, False)
            return ds

    def outputProcessor(self, modelOut, context):
        # print(modelOut)
        data = self.tickerData[context['ticker']][self.targetFeature]
        factor = max(data) - min(data)
        out = (modelOut * factor) + min(data)
        nOut = np.zeros(shape=(len(out), self.forecast))
        for i, row in enumerate(out):
            nOut[i] = row[-self.forecast:].reshape(-self.forecast,)
        return nOut

    def getTrainingData(self, shuffle=True):
        context = {}
        context["isTrain"] = True
        lenTrain = 0
        lenValid = 0
        trainDS = None
        validDS = None
        for i, ticker in enumerate(self.tickers):
            context["ticker"] = ticker
            data = self.tickerData[ticker].copy()
            splitIndex = math.floor(self.validationSplit * len(data))
            lenTrain += len(data[:splitIndex])
            lenValid += len(data[splitIndex:])
            if i == 0:
                trainDS = self.inputProcessor(data[:splitIndex], context)
                validDS = self.inputProcessor(data[splitIndex:], context)
            else:
                tmpTrain = self.inputProcessor(data[:splitIndex], context)
                tmpValid = self.inputProcessor(data[splitIndex:], context)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        if shuffle:
            trainDS.shuffle(lenTrain)
            validDS.shuffle(lenValid)
        trainDS = trainDS.batch(self.batchSize).prefetch(1)
        validDS = validDS.batch(self.batchSize).prefetch(1)
        return trainDS, validDS

    def convertToWindows(self, data, isTrain):
        data = data.values
        dataset = DS.Dataset.from_tensor_slices(data)
        if isTrain:
            dataset = dataset.window(self.lookBack + self.forecast,
                                     shift=1,
                                     drop_remainder=True)
            dataset = dataset.flat_map(
                lambda w: w.batch(self.lookBack + self.forecast))
            if self.isSeq2Seq:
                dataset = dataset.map(lambda w: (
                    w[:-self.forecast], w[self.forecast:, self.yInd]))
            else:
                dataset = dataset.map(lambda w: (
                    w[:-self.forecast], w[-self.forecast:, self.yInd]))
            return dataset
        else:
            dataset = dataset.window(self.lookBack, shift=self.forecast,
                                     drop_remainder=True)
            dataset = dataset.flat_map(lambda w: w.batch(self.lookBack))
            dataset = dataset.batch(1).prefetch(1)
            return dataset
