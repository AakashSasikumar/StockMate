from DataStore.APIInterface import AlphaVantage
import CONFIG
from DataStore import Indices as indi
import os
from multiprocessing import Process


def init():
    global source, indices
    source = AlphaVantage(CONFIG.ALPHA_VANTAGE_API)
    indi.init()
    indices = indi.getIndices()


def saveDailyAdjusted(index=None, ticker=None):
    if ticker and not index:
        # only ticker
        data = source.retreiveDailyAdjusted(ticker)
        with open("DataStore/StockData/{}.csv".format(ticker), "a+") as f:
            f.write(data)
    elif index and not ticker:
        # only index
        if isinstance(index, list):
            for item in index:
                print(item)
                indexName = list(item.keys())[0]
                print(indexName)
                for ticker in item[indexName]["constituents"]:
                    print(ticker)
                    if ticker + ".csv" in os.listdir("DataStore/StockData/"):
                        continue
                    data = source.retreiveDailyAdjusted(ticker)
                    with open("DataStore/StockData/{}.csv".format(ticker), "a+") as f:
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
    for category in indices["type"]:
        # proc = Process(target=saveDailyAdjusted, args=[indices["type"][category]])
        # proc.start()
        # proc.join()
        print(category)
        saveDailyAdjusted(index=indices["type"][category])
