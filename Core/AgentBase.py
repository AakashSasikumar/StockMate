import numpy as np


class AgentBase:
    def __init__(self, initialMoney):
        self.dataProcessor = None
        self.actions = self.initActions()

        self.orderBook = []
        self.initialMoney = initialMoney
        self.money = initialMoney
        self.profit = 0

    def assignDataProcessor(self, dataProcessor):
        self.dataProcessor = dataProcessor

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

    def getAction(self, state):
        processedInput = self.dataProcessor.inputProcessor(state, None)
        modelOut = self.model.predict(processedInput)
        processedOutput = self.dataProcessor.outputProcessor(modelOut, None)
        return np.argmax(processedOutput)

    def handleAction(self, action, price):
        if action == self.ACTION_BUY and self.money >= price:
            self.orderBook.append(price)
            self.money -= price
        elif action == self.ACTION_SELL and len(self.orderBook) > 0:
            lastPrice = self.orderBook.pop()
            self.profit += price - lastPrice
            self.money += price
