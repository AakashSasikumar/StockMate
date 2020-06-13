from DataStore import Indices as Indices
import os
import sys
import importlib
import inspect
from tensorflow import keras
import json


def init():
    """Method to initialize this module

    Retrieves all keras models and their parameters.
    """
    sys.path.append("/Users/aakashsasikumar/Documents/Code/Python/StockMate/")
    kerasLayers = getKerasLayers()
    initForecasterDetails(kerasLayers)


def getKerasLayers():
    """Method to get all layers in Keras

    This method is used to filter out false detections when getting
    model details. The initForecasterDetails() method will skip classes
    that come under keras layers.
    """
    kerasLayers = inspect.getmembers(keras.layers, inspect.isclass)
    kerasLayers = [layer[0] for layer in kerasLayers]
    return kerasLayers


def initForecasterDetails(kerasLayers, location="Models/Forecasters",
                          skipList=["NaiveModel", "BasicRegressor",
                                    "DenseRegressor", "RegressorBase"]):
    """Method to get all the details of defined models

    The details include:
        description: A multiline description of the model that is parsed from
                     the model's documentation
        params: A list of all class inputs required by the model
        moduleLoc: A string of the exact location of the model

    Parameters
    ----------
    kerasLayers: list
        A list of all the layers defined in keras.layers
    location: str
        The location in which all the models are defined
    skipList: list
        A list of all the models that are to be skipped by this method
    """
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
                    allForecasters[forecasterName]["moduleLoc"] = moduleLoc


def getForecasterDescription(docString):
    """Method to parse the multiline description of the model

    Parameters
    ----------
    docString: str
        The entire documentation multistring of the model

    Returns
    -------
    description: str
        A concise description of the model
    """
    descriptionRaw = docString.split("\n\n")[1]
    descriptionLines = [line.strip() for line in descriptionRaw.splitlines()]
    return " ".join(descriptionLines)


def getForecasterParams(docString, skipList=["model"]):
    """Method to parse the parameters of the model

    This method uses the documentation convention of this project
    to parse the parameters

    Parameters
    ----------
    docString: str
        The entire documentation multistring of the model
    skipList: list
        List of parametes to be skipped

    Returns
    -------
    params: list
        A list of all params required by the model
    """
    paramsRaw = docString.split("\n\n")[-1]
    params = []
    for line in paramsRaw.splitlines():
        if ":" in line:
            tmp = line.strip()
            param = tmp.split(":")[0]
            if param not in skipList:
                params.append(param)
    return params


def getAllIndicesAndConstituents():
    """Method to get all the indices all its constituents
    """
    ind = Indices.NSEIndices()
    return ind.getIndices()["type"]


def getAllFeatures():
    # currently only these are supported
    features = ["open", "high", "low", "close", "volume"]
    return features


def getTelegramAPIKey():
    if "telegramAPIData.json" not in os.listdir():
        return False
    else:
        with open("telegramAPIData.json") as f:
            return json.load(f)["apiKey"]


if __name__ == "__main__":
    init()
