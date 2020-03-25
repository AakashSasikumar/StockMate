import tensorflow as tf
from tensorflow import keras
import os
import datetime
import pickle


class RegressorBase():
    def getProjectRoot(self):
        """Returns the root directory of the folder

        This function is used to make sure that this module works even
        if used by other modules in other directories.

        Returns
        -------
        currentPath:
            The root path of the project
        """
        currentPath = os.getcwd()
        while(True):
            if "DataStore" in os.listdir(currentPath):
                break
            currentPath = "/".join(currentPath.split("/")[:-1])
        return currentPath + "/"

    def saveModel(self, savePath="DataStore/SavedModels/Forecasters/"):
        # TODO:
        # Handle case where a model hasn't been trained yet
        projectRoot = self.getProjectRoot()
        ds = projectRoot + "DataStore/"
        savePath = projectRoot + savePath
        if "SavedModels" not in os.listdir(ds):
            os.mkdir(ds + "SavedModels")
            os.mkdir(ds + "SavedModels/Agents")
            os.mkdir(ds + "SavedModels/Forecasters")
        modelName = self.__class__.__name__
        if modelName not in os.listdir(savePath):
            os.mkdir(savePath + modelName)
        savePath = savePath + modelName

        # TODO:
        # Handle the case where a model already exists for the same date
        saveDateTime = str(datetime.datetime.now())[:-10].replace(" ", "@")

        savePath = "{}/{}/".format(savePath, saveDateTime)
        os.mkdir(savePath)
        with open(savePath + "modelSummary.txt", "w+") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(savePath + "modelConfig.json", "w+") as f:
            f.write(self.model.to_json())
        with open(savePath + "history.pickle", "wb+") as f:
            pickle.dump(self.history, f)
        self.model.save(savePath + "model", save_format="tf")

    def loadModel(self, savePath="DataStore/SavedModels/Forecasters/", latest=True, file=None):
        # TODO: handle case where models are not there
        if latest:
            modelName = self.__class__.__name__
            savePath += modelName
            latestSave = sorted(os.listdir(savePath),
                                key=lambda x: self.getDatetime(x))[-1]
            savePath = "{}/{}".format(savePath, latestSave)
            self.model = keras.models.load(savePath)
        else:
            # TODO: Implement way to load specific date
            pass

    def getDatetime(self, date):
        dt = datetime.datetime.strptime(date, "%Y-%m-%d@%H:%M")
        return dt

    def convertToWindows(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.window(self.lookBack + self.forecast,
                                 shift=self.forecast, drop_remainder=True)
        dataset = dataset.flat_map(lambda w: w.batch(self.lookBack + self.forecast))
        dataset = dataset.shuffle(len(data))
        dataset = dataset.map(lambda w: (w[:-self.forecast], w[-self.forecast:]))
        dataset = dataset.batch(self.batchSize).prefetch(1)
        return dataset

    def train(self, trainDS, validDS, epochs=1000, earlyStopping=True,
              patience=15, callbacks=[]):
        keras.backend.clear_session()
        callbacks = []
        if earlyStopping:
            callback = keras.callbacks.EarlyStopping(patience=patience)
            callbacks.append(callback)
        history = self.model.fit(trainDS, epochs=epochs, validation_data=validDS,
                                 callbacks=callbacks)
        self.history = history.history
