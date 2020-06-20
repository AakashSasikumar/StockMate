from flask import Flask
from flask import render_template
from flask import request
import json

from Utils import UIInitializer as uint
from Utils import RequestHandler as urh
import Core.TelegramBot.Bot as tbot
from threading import Thread
import os


app = Flask(__name__, template_folder="UI/templates",
            static_folder="UI/static")

websiteName = "StockMate"


def init():
    uint.init()
    # if "telegramAPIData.json" in os.listdir():
    #     tbot.init()
    #     tbot.startListening()


@app.route("/")
def index():
    return render_template("index.html", title=websiteName,
                           numForecasters=len(uint.getAllSavedForecasters()),
                           numAgents=len(uint.getAllSavedAgents()))


@app.route("/myForecasters")
def myForecasters():
    pageName = "My Forecasters"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("myForecasters.html", title=title,
                           pageName=pageName,
                           savedModels=uint.getAllSavedForecasters())


@app.route("/viewModel", methods=["GET"])
def viewForecasters():
    modelLoc = request.args["selectedModel"]
    modelType = request.args["type"]
    modelName = modelLoc.split("/")[-1]
    if modelType == "agent":
        tickers = uint.getAllSavedAgents()[modelName]["tickers"]
    elif modelType == "forecaster":
        tickers = uint.getAllSavedForecasters()[modelName]["tickers"]

    title = "{}-{}".format(websiteName, modelName)
    return render_template("viewModel.html", title=title,
                           modelName=modelName,
                           modelLoc=modelLoc,
                           tickers=tickers,
                           modelType=modelType)


@app.route("/getPlot", methods=["POST"])
def getPlot():
    ticker = request.json["ticker"]
    modelLoc = request.json["modelLoc"]
    plotType = request.json["plotType"]
    numDays = request.json["numDays"]
    modelType = request.json["type"]
    if modelType == "forecaster":
        figure = urh.getForecasterPlot(modelLoc, ticker,
                                       plotType, numDays)
    elif modelType == "agent":
        figure = urh.getAgentPlot(modelLoc, ticker,
                                  plotType, numDays)
    return figure


@app.route("/createForecastersPage")
def createForecastersPage():
    pageName = "Create Forecasters"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("createForecasters.html", title=title,
                           pageName=pageName,
                           allModels=uint.allForecasters,
                           indices=uint.getAllIndicesAndConstituents(),
                           allFeatures=uint.getAllFeatures())


@app.route("/createForecaster", methods=["POST"])
def createForecaster():
    modelData = request.json
    Thread(urh.createForecaster(modelData)).start()
    return (json.dumps({'success': True}), 200,
            {'ContentType': 'application/json'})


@app.route("/myAgents")
def myAgents():
    pageName = "My Agents"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("myAgents.html", title=title,
                           pageName=pageName,
                           savedModels=uint.getAllSavedAgents())


@app.route("/createAgents")
def createAgents():
    pageName = "Create Agents"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("createAgents.html", title=title, pageName=pageName)


@app.route("/toggleAgentSubscription", methods=["POST"])
def toggleAgentSubscription():
    modelData = request.json
    urh.toggleAgentSubscription(modelData)
    return (json.dumps({'success': True}), 200,
            {'ContentType': 'application/json'})


@app.route("/subscriptions")
def subscriptions():
    pageName = "Create Forecasters"
    title = "{}-{}".format(websiteName, pageName)

    return render_template("subscriptions.html", title=title,
                           pageName=pageName,
                           apiKey=uint.getTelegramAPIKey())


@app.route("/botFatherInstructions")
def botCreation():
    pageName = "Bot Creation"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("botFatherInstructions.html", title=title,
                           pageName=pageName)


@app.route("/submitTelegramAPIKey", methods=["POST"])
def submitTelegramAPIKey():
    apiKey = request.json["apiKey"]
    urh.saveTelegramAPIKey(apiKey)
    return (json.dumps({'success': True}), 200,
            {'ContentType': 'application/json'})


@app.route("/resetTelegramRoot", methods=["POST"])
def resetRoot():
    urh.resetTelegramRoot()
    return (json.dumps({'success': True}), 200,
            {'ContentType': 'application/json'})


@app.errorhandler(404)
def pageNotFound(e):
    title = "404: Page Not Found"
    return render_template("404.html", title=title)


if __name__ == '__main__':
    init()
    app.run(debug=True)
else:
    init()
