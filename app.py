from flask import Flask
from flask import render_template
from flask import request
import json

from Utils import UIInitializer as uint
from Utils import RequestHandler as urh
import Core.TelegramBot.Bot as tbot
from threading import Thread


app = Flask(__name__, template_folder="UI/templates",
            static_folder="UI/static")

websiteName = "StockMate"


def init():
    uint.init()
    # tbot.init()
    # tbot.startListening()


@app.route("/")
def index():
    return render_template("index.html", title=websiteName)


@app.route("/myForecasters")
def myForecasters():
    pageName = "My Forecasters"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("myForecasters.html", title=title,
                           pageName=pageName,
                           savedModels=uint.getAllSavedForecasters())


@app.route("/viewForecaster", methods=["GET"])
def viewForecasters():
    modelLoc = request.args["selectedModel"]
    tickers = urh.getTickers(modelLoc)
    title = "{}-{}".format(websiteName, modelLoc.split("/")[-1])
    return render_template("viewForecaster.html", title=title,
                           modelName=modelLoc.split("/")[-1],
                           modelLoc=modelLoc,
                           tickers=tickers)


@app.route("/getPlot", methods=["POST"])
def getPlot():
    ticker = request.json["ticker"]
    modelLoc = request.json["modelLoc"]
    rawPlot, layout = urh.getTickerPlot(modelLoc, ticker, uint.allForecasters)
    return json.dumps({"data": rawPlot, "layout": layout})


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
                           pageName=pageName)


@app.route("/createAgents")
def createAgents():
    pageName = "Create Agents"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("createAgents.html", title=title, pageName=pageName)


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
    app.run(debug=False)
else:
    init()
