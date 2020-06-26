import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import yfinance as yf


ticker = "INDUSINDBK"


class TradeDataset(Dataset):

    def __init__(self, ticker, lookBack=30):
        self.data = yf.Ticker("{}.NS".format(ticker)).history("max").reset_index()
        self.data = self.data.sort_index()
        self.lookBack = lookBack
        self.createDataset()

    def createDataset(self):
        tmp = self.data.copy()["Close"]
        tmp = tmp.diff(1).dropna().values
        shape = tmp.shape[:-1] + (tmp.shape[-1] - self.lookBack + 1, self.lookBack)
        strides = tmp.strides + (tmp.strides[-1],)
        self.dataset = np.lib.stride_tricks.as_strided(tmp, shape=shape, strides=strides)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index]).float()


dataset = TradeDataset(ticker=ticker)


class Model(nn.Module):

    def __init__(self, lookBack, actionSize):
        super(Model, self).__init__()
        self.lookBack = lookBack
        self.actionSize = actionSize
        self.hidden = nn.Linear(in_features=lookBack, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=actionSize)

    def forward(self, X):
        X = F.relu(self.hidden(X))
        X = self.out(X)
        return X


model = Model(lookBack=30, actionSize=3)
lr = 0.001
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=lr)


class DQN:
    def __init__(self, lookBack=30,
                 gamma=0.95, epsilon=0.5,
                 epsilonMin=0.01, epsilonDecay=0.99,
                 learningRate=0.001, batchSize=32,
                 money=10000):
        self.lookBack = lookBack
        self.initialMoney = money
        self.actionSize = 3

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.batchSize = batchSize

        self.orderBook = []
        self.memory = []
        self.history = {}

    def assignModel(self, model):
        self.model = model

    def getAction(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.model(state).detach().numpy()[0])

    def handleAction(self, action, currentPrice):
        # buy action
        if action == 0 and self.money >= currentPrice:
            self.orderBook.append(currentPrice)
            self.money -= currentPrice
        # sell action
        elif action == 1 and len(self.orderBook) > 0:
            lastBuyPrice = self.orderBook.pop()
            self.profit += currentPrice - lastBuyPrice
            self.money += currentPrice
        # we will not do anything for hold action

    def getReward(self, action, currentPrice):
        self.handleAction(action, currentPrice)
        return (self.money - self.initialMoney)/self.initialMoney

    def trainModel(self):
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = torch.empty((self.batchSize, self.lookBack))
            Y = torch.empty((self.batchSize, self.actionSize))
            states = torch.empty(len(batchData), self.lookBack)
            newStates = torch.empty(len(batchData), self.lookBack)
            for i, item in enumerate(batchData):
                states[i] = item[0]
                newStates[i] = item[3]
            # states = [item[0] for item in batchData]
            # newStates = [item[3] for item in batchData]
            Q = self.model(states)
            QNext = self.model(newStates)
            for i in range(len(batchData)):
                state, action, reward, nextState = batchData[i]
                target = Q[i]
                target[action] = reward
                target[action] += self.gamma * torch.max(QNext[i])
                X[i, :] = state
                Y[i, :] = target
            self.updateWeights(X, Y)
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def updateWeights(self, X, Y):
        optimizer.zero_grad()
        out = self.model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

    def train(self, epochs=200, logFreq=1):
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm(range(self.lookBack, len(dataset.data)-1)):
                currentPrice = dataset.data["Close"].values[timeStep]
                currentState = dataset[timeStep-self.lookBack]
                nextState = dataset[timeStep-self.lookBack+1]

                action = self.getAction(currentState.reshape(1, -1))

                reward = self.getReward(action, currentPrice)

                self.memory.append((currentState, action, reward, nextState))

                self.trainModel()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                print(logStr.format(epoch, epochs, self.profit, self.money))


def trainAgent():
    test = DQN()
    test.assignModel(model)
    test.train(1)


if __name__ == "__main__":
    trainAgent()
