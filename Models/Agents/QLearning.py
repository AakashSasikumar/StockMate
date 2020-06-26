import tensorflow.compat.v1 as tf
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
                 batchSize=32):
        super().__init__(initialMoney)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = epsilonDecay
        self.memorySize = memorySize
        self.batchSize = batchSize

        self.memory = []
        self.history = {"epochs": [], "profit": [], "total_money": []}

    def buildModel(self, learningRate=1e-5):
        tf.reset_default_graph()
        self.session = tf.Session()

        self.input = tf.placeholder(name="stateInput", dtype=tf.float32,
                                    shape=[None, self.lookBack-1])
        self.target = tf.placeholder(name="target", dtype=tf.float32,
                                     shape=[None, self.actionSize])

        inputLayer = tf.layers.dense(self.input, 256, activation=tf.nn.relu)
        self.modelOutput = tf.layers.dense(inputLayer, self.actionSize)

        self.costFunc = tf.reduce_mean(tf.square(self.target - self.modelOutput))
        self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(self.costFunc)
        self.session.run(tf.global_variables_initializer())

    def getAction(self, state, context=None):
        modelOut = self.session.run(self.modelOutput,
                                    feed_dict={self.input: state.reshape(1, -1)})
        action = self.dataProcessor.outputProcessor(modelOut, None)
        return action

    def getReward(self, action, price):
        self.handleAction(action, price)
        return (self.money - self.initialMoney) / self.initialMoney

    def updateWeights(self):
        """Method to update the weights of the Q network
        """
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = np.empty((self.batchSize, self.lookBack-1))
            Y = np.empty((self.batchSize, self.actionSize))
            states = np.array([item[0] for item in batchData])
            newStates = np.array([item[3] for item in batchData])
            Q = self.session.run(self.modelOutput,
                                 feed_dict={self.input: states})
            QNext = self.session.run(self.modelOutput,
                                     feed_dict={self.input: newStates})
            for i in range(len(batchData)):
                state, action, reward, nextState, profitable = batchData[i]
                target = Q[i]
                target[action] = reward
                if profitable:
                    target[action] += self.gamma * np.amax(QNext[i])

                X[i] = state
                Y[i] = target
            cost, _ = self.session.run([self.costFunc, self.optimizer],
                                       feed_dict={self.input: X,
                                                  self.target: Y})
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
                                           self.lookBack]
                if random.random() <= self.epsilon:
                    action = random.randrange(self.actionSize)
                else:
                    action = self.getAction(currentState)
                reward = self.getReward(action, currentPrice)
                profitable = self.money > self.initialMoney
                self.memory.append((currentState, action, reward,
                                    nextState, profitable))
                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                self.history["epochs"].append(epoch)
                self.history["profit"].append(self.profit)
                self.history["total_money"].append(self.money)
                print(logStr.format(epoch+1, epochs, self.profit, self.money))
