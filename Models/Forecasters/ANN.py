from Models.KerasBase import RegressorBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import math


class BasicRegressor(RegressorBase):
    def __init__(self, lookBack=4, forecast=1,
                 batchSize=32, loadLatest=False):
        self.lookBack = lookBack
        self.forecast = forecast
        self.model = None
        self.batchSize = batchSize
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=1e-3):
        model = keras.models.Sequential()
        model.add(Dense(self.forecast, input_shape=[self.lookBack]))
        optimizer = keras.optimizers.Adam(lr=learningRate)
        model.compile(loss="mean_squared_error", optimizer=optimizer,
                      metrics=['mse'])
        self.model = model

    def convertToWindowedDS(self, data):
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast,
                       shift=self.forecast, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.shuffle(len(data))
        ds = ds.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        ds = ds.batch(self.batchSize).prefetch(1)
        return ds


class DenseRegressor(RegressorBase):
    def __init__(self, lookBack=4, forecast=1, loadLatest=False):
        self.lookBack = lookBack
        self.forecast = forecast
        self.model = None
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=1e-3):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(self.lookBack,
                                     input_shape=[self.lookBack]))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*4))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast))
        optimizer = keras.optimizers.RMSprop(lr=learningRate)
    #     optimizer = keras.optimizers.Adam(lr=learningRate)
    #     model.compile(loss="mean_squared_error", optimizer=optimizer,
    #                   metrics=['mse'])
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer,
                      metrics=['mse'])

        self.model = model

    def convertToWindows(self, data):
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack + self.forecast, shift=self.forecast,
                       drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        ds = ds.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        return ds

    def convertToWindowedDS(self, data, splitRatio=0.7, batchSize=32):
        lenTrain = 0
        lenValid = 0
        for i, ticker in enumerate(list(data.columns)):
            values = data[ticker].values
            splitInd = math.floor(splitRatio * len(values))
            train = values[:splitInd]
            valid = values[splitInd:]
            lenTrain += len(train)
            lenValid += len(valid)

            if i == 0:
                trainDS = self.convertToWindows(train)
                validDS = self.convertToWindows(valid)
            else:
                tmpTrain = self.convertToWindows(train)
                tmpValid = self.convertToWindows(valid)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        trainDS = trainDS.shuffle(lenTrain)
        validDS = validDS.shuffle(lenValid)
        trainDS = trainDS.batch(batchSize).prefetch(1)
        validDS = validDS.batch(batchSize).prefetch(1)

        return trainDS, validDS
