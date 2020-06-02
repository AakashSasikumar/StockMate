from Core.DataProcessor import DataProcessor
import numpy as np


class BasicProcessor(DataProcessor):
    def __init__(self, tickers, features):
        super().__init__(tickers, features)

        self.lookBack = 30
        self.forecast = 5
        self.tickers = tickers
        self.features = features

        self.tickerData = self.getTickerData()

    def inputProcessor(self, data, context):
        close = data["close"]
        if context["isTrain"]:
            X, Y = self.convertToWindows(close, True)
            return X, Y
        else:
            X = self.convertToWindows(close, False)
            return X

    def outputProcessor(self, modelOut, context):
        return modelOut

    def convertToWindows(self, data, isTrain, batchSize=32, shuffle=False):
        data = data.values
        if isTrain:
            windowSize = self.lookBack + self.forecast
            windowData = self.splitArray(data, windowSize, dropRemaining=True)
            if shuffle:
                np.random.shuffle(windowData)

            X = [item[:self.lookBack] for item in windowData]
            Y = [item[self.lookBack:] for item in windowData]

            trainX = self.splitArray(X, batchSize)
            trainY = self.splitArray(Y, batchSize)

            # TODO:
            # Implement batching
            return trainX, trainY
        else:
            windowData = self.splitArray(data, self.lookBack)
            return windowData

    def splitArray(self, array, unitLength, dropRemaining=False):
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


if __name__ == "__main__":
    a = BasicProcessor(["INDUSINDBK"], ["open", "high", "low", "close"])
    a.processInput()
