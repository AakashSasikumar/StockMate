import json
import os
import Core.TelegramBot.Bot as tbot
import inspect
import sys
import Examples.Processors.BasicProcessors as bp
import traceback


def saveTelegramAPIKey(apiKey):
    saveDict = {"apiKey": apiKey}
    with open("telegramAPIData.json", "w+") as f:
        json.dump(saveDict, f)


def resetTelegramRoot():
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
    forecasters = inspect.getmembers(sys.modules[moduleLoc],
                                     inspect.isclass)
    for forecaster in forecasters:
        if forecaster[0] == model:
            return forecaster[1]


def buildForecaster(modelData, model):
    multiVar = buildForecasterDP(modelData)
    model.assignDataProcessor(multiVar)
    model.buildModel()
    return model


def buildForecasterDP(modelData):
    multiVar = bp.MultiVarProcessor(tickers=modelData["tickers"],
                                    features=modelData["features"],
                                    lookBack=int(modelData["lookBack"]),
                                    forecast=int(modelData["forecast"]),
                                    targetFeature=modelData["targetFeature"],
                                    isSeq2Seq=True)
    return multiVar


def trainAndSaveForecaster(model, modelName):
    model.train(validationSplit=0.8, epochs=2, batchSize=32)
    model.saveModel(modelName)
    return model.history


def afterTrainProcedure(history, modelData):
    global lastTrainedModelData

    try:
        valMSE = history["val_mse"][-1]
        trainMSE = history["mse"][-1]
        message = ("The early stop callback was triggered for {}. The final "
                   "trainMSE = {:.3} and valMSE = {:.3}. Do you want to "
                   "retrain the model? (yes/no)")
        tbot.sendMessage(message.format(modelData["modelName"], trainMSE,
                                        valMSE))
        print(tbot.retrainFilter.toggleRetrain)
        tbot.retrainFilter.toggleRetrain = True
        print(tbot.retrainFilter.toggleRetrain)
        lastTrainedModelData = modelData
    except Exception:
        message = "The early stop callback was triggered for {}"
        tbot.sendMessage(message.format(modelData["modelName"]))
        traceback.print_exc()


def retrainForecaster():
    moduleLoc = lastTrainedModelData["moduleLoc"]
    model = lastTrainedModelData["model"]
    modelName = lastTrainedModelData["modelName"]
    model = getForecasterClass(moduleLoc, model)
    model = model()
    model.loadModel(modelName)
    print("asdfasdf")

    history = trainAndSaveForecaster(model,
                                     lastTrainedModelData["modelName"])

    afterTrainProcedure(history, lastTrainedModelData)
