from flask import Flask
from flask import render_template
from flask import request
import json

from Utils import UIInitializer as uint


app = Flask(__name__, template_folder="UI/templates",
            static_folder="UI/static")

websiteName = "StockMate"


def init():
    uint.init()


@app.route("/")
def index():
    return render_template("index.html", title=websiteName)


@app.route("/myForecasters")
def myForecasters():
    pageName = "My Forecasters"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("myForecasters.html", title=title,
                           pageName=pageName)


@app.route("/createForecastersPage")
def createForecastersPage():
    pageName = "Create Forecasters"
    title = "{}-{}".format(websiteName, pageName)
    print()
    return render_template("createForecasters.html", title=title,
                           pageName=pageName,
                           allModels=uint.allForecasters,
                           allParams=uint.getUniqueForecasterParams(),
                           indices=uint.getAllIndicesAndConstituents(),
                           allFeatures=uint.getAllFeatures())


@app.route("/createForecaster", methods=["POST"])
def createForecaster():
    modelData = request.json
    # TODO
    # Write method to create the model as specified
    print(modelData)
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


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
                           pageName=pageName)


@app.route("/botFatherInstructions")
def botCreation():
    pageName = "Bot Creation"
    title = "{}-{}".format(websiteName, pageName)
    return render_template("botFatherInstructions.html", title=title,
                           pageName=pageName)


@app.errorhandler(404)
def pageNotFound(e):
    title = "404: Page Not Found"
    return render_template("404.html", title=title)


if __name__ == '__main__':
    init()
    app.run(debug=True)
