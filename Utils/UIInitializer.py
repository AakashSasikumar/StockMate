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
                          skipList=["NaiveModel", "KerasBase", "RegressorBase"]):
    global allForecasters

    allForecasters = {}

    rootImport = location.replace("/", ".")
    for forecasterType in os.listdir(location):
        if ".py" in forecasterType:
            moduleLoc = ".".join([rootImport, forecasterType[:-3]])
            forecasterFileName = importlib.import_module(moduleLoc).__name__
            forecasters = inspect.getmembers(sys.modules[forecasterFileName],
                                             inspect.isclass)
            for forecaster in forecasters:
                forecasterName = forecaster[0]
                if forecasterName not in kerasLayers and \
                   forecasterName not in skipList:

                    allForecasters[forecasterName] = {}
                    allForecasters[forecasterName]["description"] = \
                        getForecasterDescription(forecaster[1].__doc__)
                    allForecasters[forecasterName]["params"] = \
                        getForecasterParams(forecaster[1].__doc__)


def getForecasterDescription(docString):
    descriptionRaw = docString.split("\n\n")[1]
    descriptionLines = [line.strip() for line in descriptionRaw.splitlines()]
    # print(" ".join(descriptionLines))
    return " ".join(descriptionLines)


def getForecasterParams(docString, skipList=["model"]):
    paramsRaw = docString.split("\n\n")[-1]
    params = []
    for line in paramsRaw.splitlines():
        if ":" in line:
            tmp = line.strip()
            param = tmp.split(":")[0]
            if param not in skipList:
                params.append(param)
    # print(params)
    return params


def getUniqueForecasterParams():
    allParams = []
    for model in allForecasters:
        params = allForecasters[model]["params"]
        allParams.extend(params)
    return list(set(allParams))


if __name__ == "__main__":
    init()
