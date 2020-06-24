import schedule
import sys
from Utils import UIInitializer as uint
import inspect
import pandas as pd
from DataStore import APIInterface
from Core.TelegramBot import Bot as bot
import os
import json
from threading import Thread
import time
from Examples.Processors import BasicProcessors


source = APIInterface.YFinance()
allJobTags = []
stopJobs = False


class SubscriptionJob():
    """Class for creation of subscription jobs

    This class is used for creating and running subscription jobs.
    The message interval for these jobs is the same as the interval
    specified in the data processor of the agent. If the interval
    is 1D, then a message will be sent at 3:15, when the interval is
    in minutes, the message frequency will match it.

    Attributes
    ----------
    agentName: str
        The name of the agent for which the subscription is to be made
    agentSaveLoc: str
        The save location of the agent. Used for loading and inferencing
    tickerDataPath: str, optional
        The location of all the ticker data
    """
    def __init__(self, agentName, agentSaveLoc,
                 tickerDataPath="DataStore/StockData/"):
        self.agentName = agentName
        self.agentSaveLoc = agentSaveLoc
        self.tickerDataLoc = tickerDataPath

        self.agent = self.loadAgent()
        self.interval = self.agent.dataProcessor.interval.lower()

    def loadAgent(self):
        """Method to load the agent and return it

        Returns
        -------
        model: AgentBase()
            The agent model
        """
        baseClass = self.agentSaveLoc.split("/")[-2]
        moduleLoc = uint.allAgents[baseClass]["moduleLoc"]
        agents = inspect.getmembers(sys.modules[moduleLoc],
                                    inspect.isclass)
        for agent in agents:
            if agent[0] == baseClass:
                model = agent[1]()
                break
        model.loadModel(self.agentName)
        self.name = "{}.{}".format(baseClass, self.agentName)
        self.tickers = model.dataProcessor.tickers
        return model

    def task(self):
        """Defines the tasks to be done when subscribed

        The set of tasks include,
            1. Downloading and saving the new ticker data
            2. Inferencing
            3. Messaging the root about the action
        """
        self.updateStockData()
        tickerActions = {}
        for ticker in self.tickers:
            data = self.getTickerData(ticker,
                                      self.agent.dataProcessor.interval)
            context = {"ticker": ticker,
                       "isTrain": False}
            buy, sell, hold = self.agent.getAllActions(data, context)
            if buy[0] == 1:
                # buy action
                action = "Buy"
            elif sell[0] == 1:
                # sell action
                action = "Sell"
            elif hold[0] == 1:
                # hold action
                action = "Hold"
                continue
            tickerActions[ticker] = {}
            tickerActions[ticker]["action"] = action
            tickerActions[ticker]["price"] = data["Close"].values[-1]

        message = self.generateMessage(self.name, tickerActions)
        if message:
            bot.sendMessage(message)

    def generateMessage(self, agentName, tickerActions):
        """Method to generate the message to be sent to the admin

        This method forms the message in the following fashion,

            basicAgent Signal:
            __________________
            1. BUY IOC@420.69
            2. SELL ADANIPOWER@69.420

        Parameters
        ----------
        agentName: str
            The name of the agent which is taking the action
        tickerActions: dict
            The dict containing all the actions for each ticker
            the agent was trained on

        Returns
        -------
        formatterMsg: str
            The final, formatted message to be sent to the user
        """
        title = "<b>{} Signal</b>:\n{}"
        line = "_" * (len(agentName) + 8)
        title = title.format(agentName, line)
        body = ""
        message = "\n{}. <b>{}</b> {}@{}"
        if len(tickerActions.keys()) == 0:
            return None
        for i, ticker in enumerate(tickerActions):
            tickerMsg = message.format(i+1, tickerActions[ticker]["action"],
                                       ticker, tickerActions[ticker]["price"])
            body += tickerMsg
        formattedMsg = title + body
        return formattedMsg

    def updateStockData(self):
        """Method to update all the ticker the agent was trained on
        """
        if self.interval == "1d":
            for ticker in self.tickers:
                source.saveIntraDay(ticker)
        else:
            for ticker in self.tickers:
                source.saveInterDay(ticker, self.interval)

    def scheduleJob(self):
        """Method to define the inference and messaging frequency

        Each schedule is tagged with a unique tag name, so as to be able
        to stop the schedule when unsubscribed.
        """
        global allJobTags
        allJobTags.append(self.name)
        if "d" in self.interval:
            # in terms of days
            numDays = int(self.interval[:-1])
            # schedule.every(numDays).day.at("15:15").do(self.task).tag(self.name)
            schedule.every(5).seconds.do(self.task).tag(self.name)
        elif "m" in self.interval:
            # in terms of minutes
            numMinutes = int(self.interval[:-1])
            schedule.every(numMinutes).minutes.at(":15").do(self.task).tag(self.name)

    def getTickerData(self, ticker, interval):
        """Method to return the input for the model

        This method passes the raw data through the agent's dataProcessor
        and returns the proper input form for the inputProcessor

        Parameters
        ----------
        ticker: str
            The ticker symbol of the stock
        interval:
            The interval size of the data that is to be read
        """
        path = "{}/{}/{}.csv".format(self.tickerDataLoc, interval.upper(),
                                     ticker)
        df = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
        df = self.agent.dataProcessor.getFeatures(df)
        return df[-self.agent.dataProcessor.lookBack:]

    def save(self):
        """Method to save the current subscription

        This method writes all the essential parameters into a JSON
        so it can be loaded and a new instance can be created with ease.
        This method saves all the parameters required for initializing
        SubscriptionJob().

        NOTE: The save path for all jobs is fixed to DataStore/JobStore. This
              done to make loading and running of all jobs easier and seamless.
        """
        path = "DataStore/"
        if "JobStore" not in os.listdir(path):
            os.mkdir(path+"JobStore/")
        savePath = path+"JobStore/"
        jobConfig = {}
        jobConfig["agentName"] = self.agentName
        jobConfig["agentSaveLoc"] = self.agentSaveLoc
        jobConfig["tickerDataLoc"] = self.tickerDataLoc

        fileName = "{}.subscription.json"
        fileSavePath = savePath + fileName.format(self.name)
        with open(fileSavePath, "w+") as f:
            json.dump(jobConfig, f)


def loadSubscriptionJob(filePath):
    """Module level method to load a specific subscription job

    Parameters
    ----------
    filePath: str
        The path of the saved subscription job

    Returns
    -------
    job: SubscriptionJob()
        An instance of SubscriptionJob with the parameters
        specified in the saved file.
    """
    with open(filePath) as f:
        jobConfig = json.load(f)

    name = jobConfig["agentName"]
    agentSaveLoc = jobConfig["agentSaveLoc"]
    tickerDataLoc = jobConfig["tickerDataLoc"]
    job = SubscriptionJob(name, agentSaveLoc, tickerDataLoc)
    return job


def loadAllSubscriptions():
    """Method to load all saved subscriptions
    """
    global allSubscriptionJobs
    rootPath = "DataStore/JobStore/"
    allSubscriptionJobs = []
    jobs = os.listdir(rootPath)
    subscriptionJobs = []
    for job in jobs:
        if "subscription" in job:
            filePath = rootPath + job
            subscriptionJobs.append(filePath)
    for job in subscriptionJobs:
        allSubscriptionJobs.append(loadSubscriptionJob(job))


def startAllSubscriptions():
    """Method to start all loaded subscriptions
    """
    global stopJobs
    stopJobs = False
    for job in allSubscriptionJobs:
        job.scheduleJob()


def cancelAllJobs():
    """Method to cancel all currently running jobs

    This method should only be called when the program is exiting
    """
    global stopJobs
    stopJobs = True
    time.sleep(3)
    for job in allJobTags:
        schedule.clear(job)


def startAllJobs():
    """Method to start all loaded jobs
    """
    while not stopJobs:
        schedule.run_pending()
        time.sleep(1)


def deleteSubscription(name):
    """Method to delete a particular subscription

    This method stops the job from running, and deletes
    the save file from the save path.

    Parameters
    ----------
    name: str
        The unique tag name of the subscription job
    """
    schedule.clear(name)
    path = "DataStore/JobStore/"
    os.remove(path + name + ".subscription.json")


def init():
    """Method to initialize all the jobs and to get them running
    """
    loadAllSubscriptions()
    startAllSubscriptions()
    Thread(target=startAllJobs).start()
