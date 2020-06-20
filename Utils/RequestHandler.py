import json
import os
import Core.TelegramBot.Bot as tbot
import inspect
import sys
import Examples.Processors.BasicProcessors as bp
import traceback
import dill
from Utils import Plotter as plot
from Utils import UIInitializer as uint
import pandas as pd


def saveTelegramAPIKey(apiKey):
    """Method to save the telegram apiKey locally

    Parameters
    ----------
    apiKey: str
        The telegram bot api key
    """
    saveDict = {"apiKey": apiKey}
    with open("telegramAPIData.json", "w+") as f:
        json.dump(saveDict, f)


def resetTelegramRoot():
    """Method to reset the root user of StockMate

    This method will delete the chatID of the root user, so as
    to be able to reset it.
    """
    if "telegramAPIData.json" not in os.listdir():
        return

    with open("telegramAPIData.json") as f:
        apiData = json.load(f)

    if "rootID" not in apiData.keys():
        return

    # only saves the apiKey, and effectively removing rootID
    saveTelegramAPIKey(apiData["apiKey"])
    tbot.restRoot()


def createForecaster(modelData):
    """Method to create the forecaster specified by the UI

    Parameters
    ----------
    modelData: dict
        The details of the forecaster as specified from
        the UI
    """
    try:
        model = getForecasterClass(modelData["moduleLoc"],
                                   modelData["model"])
    except Exception:
        # TODO: Implementing a logger
        message = "There was an error creating {}. Please check logs"
        tbot.sendMessage(message.format(modelData["modelName"]))
        traceback.print_exc()
        return

    model = model()

    try:
        model = buildForecaster(modelData, model)
    except Exception:
        # TODO: Implement logger
        message = "There was an error creating {}. Please check logs"
        tbot.sendMessage(message.format(modelData["modelName"]))
        traceback.print_exc()
        return

    tbot.sendMessage("{} is now training".format(modelData["modelName"]))
    try:
        history = trainAndSaveForecaster(model, modelData["modelName"])
    except Exception:
        message = ("There was an error while training {}. Please check"
                   "logs.")
        tbot.sendMessage(message.format(modelData["modelName"]))
        traceback.print_exc()
        return

    afterTrainProcedure(history, modelData)


def getForecasterClass(moduleLoc, model):
    """Method to return the Object of the specified model

    Parameters
    ----------
    moduleLoc: str
        The file in which the model is present
    model: str
        The name of the model

    Returns
    -------
    model: class
        The class of the specified model
    """
    forecasters = inspect.getmembers(sys.modules[moduleLoc],
                                     inspect.isclass)
    for forecaster in forecasters:
        if forecaster[0] == model:
            return forecaster[1]


def buildForecaster(modelData, model):
    """Method to assign the dataprocessor and build the model

    Parameters
    ----------
    modelData: dict
        The model details as specified in the UI
    model:
        The instantiated object of the specified model

    Returns
    -------
    model: keras.model
        The compiled keras model
    """
    multiVar = buildForecasterDP(modelData)
    model.assignDataProcessor(multiVar)
    model.buildModel()
    return model


def buildForecasterDP(modelData):
    """Method to build the specified data processor

    Parameters
    ----------
    modelData: dict
        The model parameters as specified in the UI

    Returns
    -------
    multiVar: Core.DataProcessor()
        The instantiated object of DataProcessor
    """
    multiVar = bp.MultiVarProcessor(tickers=modelData["tickers"],
                                    features=modelData["features"],
                                    lookBack=int(modelData["lookBack"]),
                                    forecast=int(modelData["forecast"]),
                                    targetFeature=modelData["targetFeature"],
                                    isSeq2Seq=True)
    return multiVar


def trainAndSaveForecaster(model, modelName):
    """Method to train and save the specified forecasters

    Parameters
    ----------
    model: keras.model
        The compiled model
    modelName:
        The name give to the model by the user form the UI.
        The name under which this model is to be saved.

    Returns
    -------
    model.history: dict
        The values of the specified metrics after training.
        Metrics may include MSE, Loss etc,...
    """
    model.train(validationSplit=0.8, epochs=5000, batchSize=32)
    model.saveModel(modelName)
    return model.history


def afterTrainProcedure(history, modelData):
    """Method to ask user if the model is to be retrained

    Sometimes when the callback is triggered, the model wouldn't have really
    reached the optimum. It may be beneficial to train once more till the
    call back is triggered again.

    Parameters
    ----------
    history: dict
        The values of the specified metrics after training.
        Metrics may include MSE, Loss etc,...
    modelData: dict
        The model parameters as specified in the UI
    """
    global lastTrainedModelData

    try:
        if "mse" in history.keys():
            key = "mse"
        else:
            key = "mean_squared_error"
        valMSE = history["val_{}".format(key)][-1]
        trainMSE = history[key][-1]
        message = ("The early stop callback was triggered for {}. The final "
                   "trainMSE = {:.3} and valMSE = {:.3}. Do you want to "
                   "retrain the model? (yes/no)")
        tbot.sendMessage(message.format(modelData["modelName"], trainMSE,
                                        valMSE))
        tbot.retrainFilter.toggleRetrain = True
        lastTrainedModelData = modelData
    except Exception:
        message = "The early stop callback was triggered for {}"
        tbot.sendMessage(message.format(modelData["modelName"]))
        traceback.print_exc()


def retrainForecaster():
    """Method to retrain the model
    """
    moduleLoc = lastTrainedModelData["moduleLoc"]
    model = lastTrainedModelData["model"]
    modelName = lastTrainedModelData["modelName"]
    model = getForecasterClass(moduleLoc, model)
    model = model()
    model.loadModel(modelName)
    tbot.sendMessage("Retraining {}".format(modelName))
    history = trainAndSaveForecaster(model,
                                     lastTrainedModelData["modelName"])

    afterTrainProcedure(history, lastTrainedModelData)


def getTickers(modelLoc):
    """Method to retrieve all the tickers for a model

    This method will read in the dataprocessor for the model and read the
    tickers specified during its initialization.

    Parameters
    ----------
    modelLoc:
        The location at which this model is present

    Returns
    -------
    tickers: list
        A list of all the tickers used to train this model
    """

    with open(modelLoc+"/dataProcessor.dill", "rb") as f:
        dataProc = dill.load(f)
    return dataProc.tickers


def getTickerPlot(modelLoc, ticker, plotType, numDays):
    """Method to get the plotly plots for the ticker

    This method plots the entire raw data for the ticker as well
    as the predictions.

    Parameters
    ----------
    modelLoc: str
        The save location of the model specified
    ticker: str
        The ticker symbol which is to be plotted. This is used as the title
        of the plot
    plotType: str
        The type of plot that is to be plotted
    numDays: str
        The number of days used for the model prediction

    Returns
    -------
    figure: str
        The plotly figure encoded into a JSON format
    """
    modelName = modelLoc.split("/")[-2]
    modelSaveName = modelLoc.split("/")[-1]
    modelLoc = uint.allForecasters[modelName]["moduleLoc"]

    numDays = int(numDays)

    model = getForecasterClass(modelLoc, modelName)
    model = model()
    model.loadModel(modelSaveName)

    allData = getTickerData(ticker)
    allData = model.dataProcessor.getFeatures(allData)
    targetFeature = model.dataProcessor.allFeatures[model.dataProcessor.yInd]
    context = {"isTrain": False,
               "ticker": ticker}

    if numDays < model.dataProcessor.lookBack and numDays != -1:
        message = ("Cannot predict for numDays={} when model's"
                   " lookBack={}")
        return {"error": message.format(numDays,
                                        model.dataProcessor.lookBack)}
    if numDays == -1 or numDays >= len(allData):
        reqData = allData
    else:
        numDays = int(numDays)
        reqData = allData[-numDays:]
    prediction = model.makePredictions(reqData, context)
    figure = plot.getModelPredictionFigure(ticker, allData, prediction,
                                           targetFeature.capitalize(),
                                           plotType)
    return figure


def getTickerData(ticker):
    """Method to retrieve the latest raw ticker

    Parameters
    ----------
    ticker: str
        The ticker for which the raw data is to be retrieved

    Returns
    -------
    df: pandas.DataFrame
        The raw data of the ticker
    """
    df = pd.read_csv("DataStore/StockData/{}.csv".format(ticker),
                     index_col="Date", parse_dates=["Date"])
    return df


def toggleAgentSubscription(modelData):
    """Method to subscribe and unsubscribe to agents

    Parameters
    ----------
    modelData: dict
        The specifications of the agent
    """

    template = "{}/{}"
    with open(template.format(modelData["savePath"], "AgentInfo.json")) as f:
        agentInfo = json.load(f)
    if modelData["subscribe"]:
        # TODO: Implement subscribe methods
        agentInfo["subscribed"] = modelData["subscribe"]
    else:
        # TODO: Implement unsubscribe methods
        agentInfo["subscribed"] = modelData["subscribe"]

    with open(template.format(modelData["savePath"], "AgentInfo.json"), "w+") as f:
        json.dump(agentInfo, f)
