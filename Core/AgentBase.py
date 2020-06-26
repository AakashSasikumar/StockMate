import os
import pickle
import dill
import tensorflow.compat.v1 as tf
import datetime
import json


class AgentBase:
    """The base class for all agents

    All agents created must have a name their model's output layer
    `modelOutput`.

    Attributes
    ----------
    dataProcessor: DataProcessor()
        The dataprocessor assigned to this model
    orderBook: list
        The list of actions that were taken by the user.
        This is used to keep track of the money remaning
    initialMoney: int
        The initial money given to the agent
    profit: int
        A temporary variable used for the train function
    money: int
        A temporary variable used to keep track of money remaining
    history: dict
        An attribute that keeps track of all the metrics logged during training
    lookBack: int
        The number of days to look back to make a decision
    tickerData: list
        The list of all tickers this agent was trained on
    allActions: list
        A list of all actions possible for agents
    actionSize: int
        The number of unique actions that can be taken

    """
    def __init__(self, initialMoney):
        self.dataProcessor = None
        self.initActions()

        self.orderBook = []
        self.initialMoney = initialMoney
        self.money = initialMoney
        self.profit = 0
        self.history = None

    def assignDataProcessor(self, dataProcessor):
        self.dataProcessor = dataProcessor
        self.lookBack = dataProcessor.lookBack
        self.tickerData = self.dataProcessor.tickerData.copy()

    def initActions(self):
        """Method to initialize all the possible actions for agents

        As of now there are three actions possible,
            1. Buy - 0
            2. Sell - 1
            3. Hold - 2

        Support for shorting will be added later
        """
        self.ACTION_BUY = 0
        self.ACTION_SELL = 1
        self.ACTION_HOLD = 2
        self.allActions = [self.ACTION_BUY, self.ACTION_SELL, self.ACTION_HOLD]
        self.actionSize = len(self.allActions)

    def getAction(self, state, context=None):
        """Method to get the action from the model given the current state

        This method uses the specified dataProcessor to get the proper action
        chosen by the model.

        Parameters
        ----------
        state: numpy.array
            The representation of the current state
        context: dict
            The context under which this action is to be taken

        Returns
        -------
        action: int
            The action chosen by the model
        """

        modelOut = self.session.run(self.modelOutput,
                                    feed_dict={self.input: state})
        action = self.dataProcessor.outputProcessor(modelOut, None)
        return action

    def getAllActions(self, data, context):
        """Method to get the actions from the model for a list of days

        This method is mainly used for the UI to get actions for a bunch of
        days together.

        Parameters
        ----------
        data: pandas.DataFrame
            The dataframe of the ticker data for which we get actions
        context: dict
            The context under which the inferencing is to be done
        Returns
        -------
        actions: list
            A list of actions taken by the model
        """
        inputData = self.dataProcessor.inputProcessor(data, context)
        buyActions = []
        sellActions = []
        holdActions = []
        for i in range(len(inputData)):
            action = self.getAction(inputData[i], context)
            if action == self.ACTION_BUY:
                buyActions.append(1)
                sellActions.append(0)
                holdActions.append(0)
            elif action == self.ACTION_SELL:
                sellActions.append(1)
                buyActions.append(0)
                holdActions.append(0)
            elif action == self.ACTION_HOLD:
                holdActions.append(1)
                buyActions.append(0)
                sellActions.append(0)
        return (buyActions, sellActions, holdActions)

    def handleAction(self, action, price):
        """Method to carry out additional tasks from model output

        This method is used to carry out any other background tasks that needs
        to do done based on the action chosen by the model. Currently, this
        method is to update the orderbook and keep a track of profits made.

        Parameters
        ----------
        action: int
            The action chosen by the model
        price: float
            The price at which the action was chosen
        """
        if action == self.ACTION_BUY and self.money >= price:
            self.orderBook.append(price)
            self.money -= price
        elif action == self.ACTION_SELL and len(self.orderBook) > 0:
            lastPrice = self.orderBook.pop()
            self.profit += price - lastPrice
            self.money += price

    def getProjectRoot(self):
        """Returns the root directory of the folder

        This function is used to make sure that this module works even
        if used by other modules in other directories.

        Returns
        -------
        currentPath:
            The root path of the project
        """
        currentPath = os.getcwd()
        while(True):
            if "DataStore" in os.listdir(currentPath):
                break
            currentPath = "/".join(currentPath.split("/")[:-1])
        return currentPath + "/"

    def saveModel(self, name,
                  savePath="DataStore/SavedModels/Agents/"):
        """Saves the model into the specified path.

        NOTE: If name already exists in the directory, calling saveModel
        will overwrite the existing saved files.

        This function writes the model and some additional details into
        the specified location.

        This method saves the following things:
        1. model
            The trained model
        2. dataProcessor
            The data processor used for this model
        3. AgentInfo.json
            A json file containing the following information
                - Tickers and features used to train the model
                - The datetime of saving
            This file is used for the UI to quickly get model information
            without having to load the dataProcessor

        Parameters
        ----------
        name: str
            An optional personal name given to the model
        savePath: str, optional
            The location which the model and additional info are to be saved

        """
        projectRoot = self.getProjectRoot()
        ds = projectRoot + "DataStore/"
        savePath = projectRoot + savePath
        if "SavedModels" not in os.listdir(ds):
            os.mkdir(ds + "SavedModels")
            os.mkdir(ds + "SavedModels/Agents")
            os.mkdir(ds + "SavedModels/Forecasters")
        modelName = self.__class__.__name__
        if modelName not in os.listdir(savePath):
            os.mkdir(savePath + modelName)
        savePath = savePath + modelName

        modelPath = "{}/{}/".format(savePath, name)
        if name not in os.listdir(savePath):
            os.mkdir(modelPath)

        savePath = modelPath
        saver = tf.train.Saver()
        saver.save(self.session, savePath + "model/model.ckpt")
        with open(savePath + "dataProcessor.dill", "wb+") as f:
            dill.dump(self.dataProcessor, f)
        if self.history:
            with open(savePath + "history.pickle", "wb+") as f:
                pickle.dump(self.history, f)

        with open(savePath + "AgentInfo.json", "w+") as f:
            writeJson = {}
            writeJson["tickers"] = self.dataProcessor.tickers
            writeJson["features"] = self.dataProcessor.features
            writeJson["savedTime"] = str(datetime.datetime.now())[:-10]
            json.dump(writeJson, f)

    def loadModel(self, name, savePath="DataStore/SavedModels/Agents/"):
        """Loads the specified model.

        This method is just for preparing the input and exception
        handling.

        Parameters
        ----------
        savePath: str, optional
            The location from which the model is to loaded
        name: str
            Used to specify a certain date. If none, loads the latest. This has
            to be the folder name of the saved model.
        """
        savePath = self.getProjectRoot() + savePath
        modelName = self.__class__.__name__
        modelPath = savePath + modelName
        if name not in os.listdir(modelPath):
            message = "Could not find {} in {}".format(name, savePath)
            raise Exception(message)
        savePath = modelPath
        savePath = "{}/{}".format(savePath, name)
        dp = self.loadDP(savePath)
        self.assignDataProcessor(dp)
        self.buildModel()
        saver = tf.train.Saver()
        saver.restore(self.session, savePath+"/model/model.ckpt")

    def loadDP(self, path):
        """Loads the model's data processor

        This method loads the model and other attributes when the
        correct path is passed.

        Parameters
        ----------
        path: str
            The path of the model

        Returns
        -------
        dataProcessor: DataProcessor()
            The data processor that was used for this model
        """
        with open(path + "/dataProcessor.dill", "rb") as f:
            dataProcessor = dill.load(f)
        return dataProcessor
