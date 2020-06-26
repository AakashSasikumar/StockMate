import numpy as np
import tensorflow.compat.v1 as tf
import random
import tqdm
import yfinance as yf

# tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.set_random_seed(1997)

ticker = "INDUSINDBK"
df_full = yf.Ticker("{}.NS".format(ticker)).history("max").reset_index()

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
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.input = tf.placeholder(
            name="stateInput", dtype=tf.float32, shape=[None, self.lookBack])
        self.target = tf.placeholder(name="target", dtype=tf.float32, shape=[
                                     None, self.actionSize])

        inputLayer = tf.layers.dense(self.input, 256, activation=tf.nn.relu)
        self.modelOut = tf.layers.dense(inputLayer, self.actionSize)

        self.costFunc = tf.reduce_mean(tf.square(self.target - self.modelOut))
        self.optimizer = tf.train.GradientDescentOptimizer(
            1e-5).minimize(self.costFunc)
        self.sess.run(tf.global_variables_initializer())

    def getAction(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actionSize)
        else:
            return np.argmax(self.sess.run(self.modelOut,
                             feed_dict={self.input: state})[0])

    def createDataset(self):
        tmp = self.data.copy()
        tmp = tmp.diff(1).dropna().values
        shape = tmp.shape[:-1] + \
            (tmp.shape[-1] - self.lookBack + 1, self.lookBack)
        strides = tmp.strides + (tmp.strides[-1],)
        self.dataset = np.lib.stride_tricks.as_strided(
            tmp, shape=shape, strides=strides)

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

    def updateWeights(self):
        if len(self.memory) >= self.batchSize:
            endIndex = len(self.memory)
            startIndex = endIndex - self.batchSize
            batchData = []
            for i in range(startIndex, endIndex):
                batchData.append(self.memory[i])
            X = np.empty((self.batchSize, self.lookBack))
            Y = np.empty((self.batchSize, self.actionSize))
            states = np.array([item[0] for item in batchData])
            newStates = np.array([item[3] for item in batchData])
            Q = self.sess.run(self.modelOut, feed_dict={self.input: states})
            QNext = self.sess.run(self.modelOut, feed_dict={
                                  self.input: newStates})
            for i in range(len(batchData)):
                state, action, reward, nextState, notProfitable = batchData[i]
                target = Q[i]
                target[action] = reward
                if not notProfitable:
                    target[action] += self.gamma * np.amax(QNext[i])

                X[i] = state
                Y[i] = target
            cost, _ = self.sess.run([self.costFunc, self.optimizer],
                                    feed_dict={self.input: X, self.target: Y})
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay

    def train(self, epochs=200, logFreq=1):
        for epoch in range(epochs):
            self.profit = 0
            self.money = self.initialMoney
            for timeStep in tqdm.tqdm(range(self.lookBack, len(self.data)-1)):
                currentPrice = data[timeStep]
                currentState = self.dataset[timeStep-self.lookBack]
                nextState = self.dataset[timeStep-self.lookBack+1]

                action = self.getAction(currentState.reshape(1, -1))

                reward = self.getReward(action, currentPrice)

                notProfitable = self.money < self.initialMoney
                self.memory.append(
                    (currentState, action, reward, nextState, notProfitable))

                self.updateWeights()
            if epoch % logFreq == 0:
                logStr = "Epoch: {}/{}  Total Profit: {}  Total Money: {}"
                print(logStr.format(epoch, epochs, self.profit, self.money))
                print(self.epsilon)

    def simulateTrade(self, money=10000):
        self.money = money
        self.profit = 0
        self.orderBook = []
        for timeStep in tqdm.tqdm(range(self.lookBack, len(self.data)-1)):
            state = self.dataset[timeStep-self.lookBack]

            action = np.argmax(self.sess.run(self.modelOut, feed_dict={
                               self.input: state.reshape(1, -1)})[0])

            self.handleAction(action, data[timeStep])
        profitPerc = (self.profit / self.initialMoney) * 100
        logStr = "Initial Amount: {}, Final Amount: {}, Profit Made: {}"
        print(logStr.format(self.initialMoney, self.money, profitPerc))


def trainAgent():
    test = DQN(data)
    test.buildModel()
    test.createDataset()
    test.train(1)


if __name__ == "__main__":
    trainAgent()
