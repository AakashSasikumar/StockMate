import tensorflow as tf
from tensorflow import keras
from Core.AgentBase import AgentBase
import numpy as np
import tqdm
import random

tf.compat.v1.disable_eager_execution()


class BasicDQN(AgentBase):
    def __init__(self, initialMoney=10000, gamma=0.95, epsilon=0.5,
                 epsilonMin=0.01, epsilonDecay=0.99, memorySize=1000,
                 trainBatchSize=32):
        super().__init__(initialMoney)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.memorySize = memorySize
        self.trainBatchSize = trainBatchSize

        self.memory = []

    def buildModel(self, learningRate=1e-5):
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256,
                                     input_shape=[self.dataProcessor.lookBack],
                                     activation="relu"))
        model.add(keras.layers.Dense(self.actionSize))

        self.optimizer = keras.optimizers.RMSprop(lr=learningRate,
                                                  epsilon=0.1,
                                                  rho=0.99)
        self.lossFunction = keras.losses.mean_squared_error
        model.compile(loss=self.lossFunction, optimizer=self.optimizer)
        self.model = model

    def getReward(self, action, price):
        self.handleAction(action, price)
        return (self.money - self.initialMoney) / self.initialMoney

    @tf.function
    def updateWeights(self):
        if len(self.memory) > self.trainBatchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.trainBatchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = tf.zeros((self.trainBatchSize, self.dataProcessor.lookBack))
            Y = tf.zeros((self.trainBatchSize, self.actionSize))
            states = np.array([item[0] for item in batchData])
            newStates = np.array([item[3] for item in batchData])
            Q = self.model(states)
            QNext = self.model(newStates)
            for i in range(len(batchData)):
                state, action, reward, nextState = batchData[i]
                target = Q[i]
                target[action] = reward
                target[action] += self.gamma * np.max(QNext[i])

                X[i] = state
                Y[i] = target

            self.model.fit(X, Y)
            if self.epsilon > self.epsilonMin:
                self.epsilonMin *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        self.trainData = self.dataProcessor.getTrainingData()
        self.rawData = self.dataProcessor.tickerData[self.dataProcessor.tickers[0]]
        self.rawData = self.rawData.values
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.dataProcessor.lookBack,
                                            len(self.rawData)-1)):
                currentPrice = self.rawData[timeStep]
                currentState = self.trainData[timeStep -
                                              self.dataProcessor.lookBack]
                nextState = self.trainData[timeStep -
                                           self.dataProcessor.lookBack+1]
                if random.random() <= self.epsilon:
                    action = random.randrange(self.actionSize)
                else:
                    action = self.getAction(currentState.reshape(1, -1))
                reward = self.getReward(action, currentPrice)
                self.memory.append((currentState, action, reward, nextState))
                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                print(logStr.format(epoch, epochs, self.profit, self.money))
