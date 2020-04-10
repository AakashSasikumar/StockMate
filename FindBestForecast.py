import pandas as pd
from DataStore import Indices
from Models.Forecasters import ANN as ann
import matplotlib.pyplot as plt


def loadData():
    global data
    ind = Indices.NSEIndices()
    category = "Broad Market Indices :"
    index = "NIFTY 50"
    tickers = ind.getIndices()['type'][category][index]['constituents']
    data = pd.DataFrame()
    for ticker in tickers:
        temp = pd.read_csv("../DataStore/StockData/{}.csv".format(ticker),
                           parse_dates=['timestamp'], index_col='timestamp')
        data[ticker] = temp['close']
    data = data.sort_index()


def trainModel():
    global data
    dr = ann.DenseRegressor()
    dr.buildModel(learningRate=1e-4)
    trainDS, validDS = dr.convertToWindowedDS(data)
    dr.train(trainDS, validDS)
    dr.saveModel()


def plotPredictions(ticker, lastN=30, tickerData=None, model=None, show=True):
    global data
    if isinstance(tickerData, pd.DataFrame):
        data = tickerData

    if model:
        dr = model
    else:
        dr = ann.DenseRegressor(loadLatest=True)

    dataPoints = data[ticker].values[-lastN:]
    forecasts = dr.makePredictions(dataPoints)
    # print(forecasts)
    plt.plot(data.index[-lastN:].date, dataPoints,
             linewidth=1, label="Actual")
    plt.plot(data.index[-lastN+dr.lookBack-1::dr.forecast].date,
             forecasts, linewidth=1, label="Predicted")
    plt.title(ticker)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def getNextDayTopPerformers(top=10, model=None, tickerData=None):
    global data
    topPotentials = []
    if not model:
        dr = ann.DenseRegressor(loadLatest=True)
    else:
        dr = model
    if isinstance(tickerData, pd.DataFrame):
        data = tickerData
    for ticker in list(data.columns):
        lastVal = data[ticker].values[-1]

        prevData = [[]]
        prevData[0] = list(data[ticker].values[-dr.lookBack:])

        nextDay = list(dr.model.predict(prevData)[0])
        pctInc = ((nextDay - lastVal) / lastVal) * 100

        topPotentials.append([ticker, pctInc, prevData, nextDay])
    topPotentials = sorted(topPotentials, key=lambda x: x[1], reverse=True)
    return topPotentials[:top]


if __name__ == "__main__":
    global data
    loadData()
    # trainModel()
    # plotPredictions("IOC")
    print(getNextDayTopPerformers(top=50))
