import pandas as pd


class DataProcessor():
    def __init__(self, tickers, features):
        self.tickers = tickers
        self.features = features

        self.loadTickerData()

    def loadTickerData(self):
        tickerData = {}
        for ticker in self.tickers:
            data = pd.read_csv("DataStore/StockData/{}.csv",
                               index_col="timestamp")
            data = self.getFeatures(data)
            tickerData[ticker] = data

    def getFeatures(self, data):
        # TODO: Implement ways to get other features
        return data
