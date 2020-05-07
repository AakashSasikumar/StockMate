# from DataStore.Indices import Indices
import json
import os
import importlib
import sys
import inspect
from tensorflow import keras


def init():
    sys.path.append("/Users/aakashsasikumar/Documents/Code/Python/StockMate/")
    # print(os.listdir("../"))
    kerasLayers = getKerasLayers()
    initForecasterDetails(kerasLayers)


def getKerasLayers():
    kerasLayers = inspect.getmembers(keras.layers, inspect.isclass)
    kerasLayers = [layer[0] for layer in kerasLayers]
    return kerasLayers


def initForecasterDetails(kerasLayers, location="Models/Forecasters",
                          skipList=["NaiveModel"]):
    global allForecasters

    allForecasters = {}

    rootImport = location.replace("/", ".")
    for forecasterType in os.listdir(location):
        if ".py" in forecasterType:
            forecasterFile = importlib.import_module(".".join([rootImport, forecasterType[:-3]]))
            forecasters = inspect.getmembers(sys.modules[forecasterFile.__name__], inspect.isclass)
            for forecaster in forecasters:
                forecasterName = forecaster[0]
                if forecasterName not in kerasLayers and "Base" not in forecasterName and forecasterName not in skipList:
                    allForecasters[forecasterName] = {}
                    allForecasters[forecasterName]["description"] = getDescription(forecaster[1].__doc__)


def getDescription(docString):
    descriptionRaw = docString.split("\n\n")[1]
    descriptionLines = [line.strip() for line in descriptionRaw.splitlines()]
    print(" ".join(descriptionLines))


if __name__ == "__main__":
    init()
