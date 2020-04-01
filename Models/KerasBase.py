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
        """Saves the model into the specified path.

        This function writes the model and some additional details into
        the specified location.

        This method saves the following things:
        1. modelSummary.txt
            A text file containing information about the model such as
            the layer wise parameters and total parameters that were trained
        2. history.pickle
            A pickle file containing the log of metrics while training the
            model.
        3. modelConfig.json
            This file contains the model configuration. This can be used
            later to replicate the model without the trained parameters

        Parameters
        ----------
        savePath: str
            The location which the model and additional info are to be saved

        """
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

    def loadModel(self, savePath="DataStore/SavedModels/Forecasters/",
                  date=False):
        """Loads a saved model

        Parameters
        ----------
        savePath: str, optional
            The location from which the model is to loaded
        date: str, optional
            Used to specify a certain date. If none, loads the latest
        """
        # TODO: handle case where models are not there
        savePath = self.getProjectRoot() + savePath
        if date:
            modelName = self.__class__.__name__
            savePath += modelName
            latestSave = sorted(os.listdir(savePath),
                                key=lambda x: self.getDatetime(x))[-1]
            savePath = "{}/{}".format(savePath, latestSave)
            self.model = keras.models.load_model(savePath+"/model")
        else:
            # TODO: Implement way to load specific date
            pass

    def getDatetime(self, date):
        """Helper function used to convert the file naming convention
        into a datetime.datetime object

        Parameters
        ----------
        date: str
            The string form of the directory

        Returns
        -------
        dt: datetime.datetime
            The corresponding datetime object
        """
        dt = datetime.datetime.strptime(date, "%Y-%m-%d@%H:%M")
        return dt

    def train(self, trainDS, validDS, epochs=1000, earlyStopping=True,
              patience=15, callbacks=[]):
        """The method to start training the model

        Parameters:
        ----------
        trainDS: tf.Dataset
            The object containing all the training data
        validDS: tf.Dataset
            The object containing all the validation data
        epochs: int, optional
            The number of epochs to train for
        earlyStopping: boolean, optional
            Specify whether early stopping based on validation
            loss is required (recommended to keep it as True)
        patience: int, optional
            The number of epochs to wait for early stopping
        callbacks: list
            custom callbacks may be specified for training
        """
        keras.backend.clear_session()
        if earlyStopping:
            callback = keras.callbacks.EarlyStopping(patience=patience)
            callbacks.append(callback)
        history = self.model.fit(trainDS, epochs=epochs,
                                 validation_data=validDS,
                                 callbacks=callbacks)
        self.history = history.history

    def makePredictions(self, data, batchSize=32):
        """Formats the data and returns the model prediction

        This method takes in the raw data, and converts it into the model's
        input specification.

        Parameters
        ----------
        data: numpy.ndarray
            The input data for prediction
        batchSize: int, optional
            The batchSize of the data

        Returns
        -------
        prediction: numpy.ndarray
            The model's prediction
        """
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.window(self.lookBack, shift=self.forecast, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.lookBack))
        ds = ds.batch(batchSize).prefetch(1)
        prediction = self.model.predict(ds)
        return prediction
