import tensorflow as tf
from tensorflow import keras
from Core.AgentBase import AgentBase
import numpy as np
import tqdm
import random

tf.compat.v1.disable_eager_execution()


class BasicDQN(AgentBase):
    """An implementation of a basic deep q learning model

    This implementation of Q learning, uses a single hidden layer
    of 256 neurons as the network. This is a univariate model that
    trains only on a single ticker.


    Attributes
    ----------
    gamma: float
        The gamma value for the Q update
    epsilon: float
        The epsilon value for exploration
    epsilonDecay: float
        A value to indicate how much the epsilon decays every epoch
    epsilonMin: float
        The least value of epsilon during training
    memorySize: int
        Value indicating how much experience is to be stored in memory
    memory: list
        A list of all previous experience tuples
    """
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
        self.history = {"epochs": [], "profit": [], "total_money": []}

    def buildModel(self, learningRate=1e-5):
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(256,
                                     input_shape=[self.dataProcessor.lookBack],
                                     activation="relu"))
        model.add(keras.layers.Dense(self.actionSize))

        optimizer = keras.optimizers.RMSprop(lr=learningRate,
                                             epsilon=0.1,
                                             rho=0.99)
        lossFunction = keras.losses.mean_squared_error
        model.compile(loss=lossFunction, optimizer=optimizer)
        self.model = model

    def getReward(self, action, price):
        self.handleAction(action, price)
        return (self.money - self.initialMoney) / self.initialMoney

    @tf.function
    def updateWeights(self):
        """Method to update the weights of the Q network
        """
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
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        self.trainData = self.dataProcessor.getTrainingData()
        self.rawData = self.tickerData[self.dataProcessor.tickers[0]]
        self.rawData = self.rawData.values
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack,
                                            len(self.rawData)-1)):
                currentPrice = self.rawData[timeStep]
                currentState = self.trainData[timeStep -
                                              self.lookBack]
                nextState = self.trainData[timeStep -
                                           self.lookBack+1]
                if random.random() <= self.epsilon:
                    action = random.randrange(self.actionSize)
                else:
                    action = self.getAction(currentState)
                reward = self.getReward(action, currentPrice)
                self.memory.append((currentState, action, reward, nextState))
                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                self.history["epochs"].append(epoch)
                self.history["profit"].append(self.profit)
                self.history["total_money"].append(self.money)
                print(logStr.format(epoch, epochs, self.profit, self.money))


class WaveNetDQN(AgentBase):
    """An implementation deep Q learning using WaveNet

    This implementation of Q learning, uses the revered
    WaveNet as the Q network. This is a univariate model that
    trains only on a single ticker.


    Attributes
    ----------
    gamma: float
        The gamma value for the Q update
    epsilon: float
        The epsilon value for exploration
    epsilonDecay: float
        A value to indicate how much the epsilon decays every epoch
    epsilonMin: float
        The least value of epsilon during training
    memorySize: int
        Value indicating how much experience is to be stored in memory
    memory: list
        A list of all previous experience tuples
    """
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
        self.history = {"epochs": [], "profit": [], "total_money": []}

    def buildModel(self, learningRate=1e-5):
        keras.backend.clear_session()
        self.numDims = len(self.dataProcessor.features)
        model = keras.models.Sequential()
        model.add(keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                  input_shape=[self.dataProcessor.lookBack, self.numDims],
                  padding="causal", activation='relu'))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.LSTM(32, return_sequences=True))
        model.add(keras.layers.Dense(1))
        if learningRate:
            optimizer = keras.optimizers.Adam(lr=learningRate)
        else:
            optimizer = keras.optimizers.Adam()
        model.compile(loss="mse", optimizer=optimizer,
                      metrics=["mse", "mae"])

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
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        self.trainData = self.dataProcessor.getTrainingData()
        self.rawData = self.tickerData[self.dataProcessor.tickers[0]]
        self.rawData = self.rawData.values
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack,
                                            len(self.rawData)-1)):
                currentPrice = self.rawData[timeStep]
                currentState = self.trainData[timeStep -
                                              self.lookBack]
                nextState = self.trainData[timeStep -
                                           self.lookBack+1]
                if random.random() <= self.epsilon:
                    action = random.randrange(self.actionSize)
                else:
                    action = self.getAction(currentState)
                reward = self.getReward(action, currentPrice)
                self.memory.append((currentState, action, reward, nextState))
                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                self.history["epochs"].append(epoch)
                self.history["profit"].append(self.profit)
                self.history["total_money"].append(self.money)
                print(logStr.format(epoch, epochs, self.profit, self.money))
                print(self.epsilon)
