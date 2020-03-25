import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import sys
sys.path.append("../")
from Models.KerasBase import RegressorBase


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

    def convertToWindows(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(self.lookBack + self.forecast,
                                 shift=self.forecast, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        dataset = dataset.shuffle(len(data))
        dataset = dataset.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        dataset = dataset.batch(self.batchSize).prefetch(1)
        return dataset


class DenseRegressor(RegressorBase):
    def __init__(self, lookBack=4, forecast=1, shuffleBuffer=2000,
                 batchSize=32, loadLatest=False):
        self.lookBack = lookBack
        self.forecast = forecast
        self.model = None
        self.shuffleBuffer = shuffleBuffer
        self.batchSize = batchSize
        if loadLatest:
            self.loadModel()

    def buildModel(self, learningRate=1e-3):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(self.lookBack, input_shape=[self.lookBack]))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*4))
        model.add(keras.layers.Dense(self.forecast*3))
        model.add(keras.layers.Dense(self.forecast*2))
        model.add(keras.layers.Dense(self.forecast))
        optimizer = keras.optimizers.RMSprop(lr=learningRate)
    #     optimizer = keras.optimizers.Adam(lr=learningRate)
    #     model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=['mse'])

        self.model = model

    def convertToWindows(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(self.lookBack + self.forecast,
                                 shift=self.forecast, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        dataset = dataset.shuffle(len(data))
        dataset = dataset.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        dataset = dataset.batch(self.batchSize).prefetch(1)
        return dataset
