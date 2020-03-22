from DataStore.APIInterface import AlphaVantage
import CONFIG
from tqdm import tqdm
from DataStore import Indices as indi
import os
import time


def init():
    global source, indices
    source = AlphaVantage(CONFIG.ALPHA_VANTAGE_API)
    indi.init()
    indices = indi.getIndices()


def saveDailyAdjusted(index=None, ticker=None):
    if ticker and not index:
        # only ticker
        data = source.getDailyAdjusted(ticker)
        with open("DataStore/StockData/{}.csv".format(ticker), "w+") as f:
            f.write(data)
    elif index and not ticker:
        # only index
        if isinstance(index, dict):
            for ticker in tqdm(index['constituents']):
                if ticker + ".csv" in os.listdir("DataStore/StockData/"):
                    continue
                data = source.getDailyAdjusted(ticker)
                with open("DataStore/StockData/{}.csv".format(ticker), "w+") as f:
                    f.write(data)
        elif isinstance(index, str):
            pass
        else:
            # raise error
            pass

    elif index and ticker:
        # combo of both
        pass
    else:
        return


if __name__ == "__main__":
    init()
    # saveDailyAdjusted(index=indices["type"]["Broad Market Indices :"]["NIFTY 50"])
    saveDailyAdjusted(ticker="TCS")

    # for i, category in enumerate(indices["type"]):
    #     print(category)
    #     saveDailyAdjusted(index=indices["type"][category])
