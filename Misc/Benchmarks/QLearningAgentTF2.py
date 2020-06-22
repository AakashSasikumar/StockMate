import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import random
import tqdm
import yfinance as yf


tf.compat.v1.disable_eager_execution()

ticker = "INDUSINDBK"
df_full = yf.Ticker("{}.NS".format(ticker)).history("max").reset_index()
# df_full = pd.read_csv("DataStore/StockData/{}.csv".format(ticker), index_col="timestamp")


df = df_full.copy()["Close"]
data = df.copy()


class DQN:
    def __init__(self, data, lookBack=30,
                 gamma=0.95, epsilon=0.5,
                 epsilonMin=0.01, epsilonDecay=0.99,
                 learningRate=0.001, batchSize=32,
                 money=10000):
        self.lookBack = lookBack
        self.initialMoney = money
        self.actionSize = 3

        self.data = data
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.batchSize = batchSize

        self.orderBook = []
        self.memory = []
        self.history = {}

    def buildModel(self):
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256, input_shape=[self.lookBack],
                                     activation="relu"))
        # model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(self.actionSize))

        self.optimizer = keras.optimizers.RMSprop(lr=self.learningRate,
                                                  epsilon=0.1,
                                                  rho=0.99)
        self.lossFunc = keras.losses.mean_squared_error
        model.compile(loss="mse", optimizer=self.optimizer)
        self.model = model

    def getAction(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.model.predict(state)[0])

    def createDataset(self):
        tmp = self.data.copy()
        tmp = tmp.diff(1).dropna().values
        shape = tmp.shape[:-1] + (tmp.shape[-1] - self.lookBack + 1, self.lookBack)
        strides = tmp.strides + (tmp.strides[-1],)
        self.dataset = np.lib.stride_tricks.as_strided(tmp, shape=shape, strides=strides)

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

    @tf.function
    def updateWeights(self):
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = tf.zeros((self.batchSize, self.lookBack))
            Y = tf.zeros((self.batchSize, self.actionSize))
            states = tf.convert_to_tensor([item[0] for item in batchData])
            newStates = tf.convert_to_tensor([item[3] for item in batchData])
            # with tf.GradientTape() as tape:
            Q = self.model(states)
            QNext = self.model(newStates)
            for i in range(len(batchData)):
                state, action, reward, nextState = batchData[i]
                target = Q[i]
                target[action] = reward
                target[action] += self.gamma * np.max(QNext[i])

                X[i] = state
                Y[i] = target
                # loss = self.lossFunc(X, Y)
                # grads = tape.gradient(loss, self.model.trainable_weights)
                # self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                # self.model.train_on_batch(X, Y)
                # self.mode.fit(X, Y, use_multiprocessing=True)
            self.model.fit(X, Y)
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        # self.optimizerFunc = self.optimizeFunc()
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack, len(self.data)-1)):
                currentPrice = data[timeStep]
                currentState = self.dataset[timeStep-self.lookBack]
                nextState = self.dataset[timeStep-self.lookBack+1]

                action = self.getAction(currentState.reshape(1, -1))
                reward = self.getReward(action, currentPrice)

                self.memory.append((currentState, action, reward, nextState))

                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                print(logStr.format(epoch, epochs, self.profit, self.money))


def trainAgent():
    test = DQN(data)
    test.createDataset()
    test.buildModel()
    test.train(2)


if __name__ == "__main__":
    trainAgent()
