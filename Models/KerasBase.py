import tensorflow as tf
from tensorflow import keras
import os
import datetime
import pickle
import json


class RegressorBase():
    """A wrapper for all keras based regression networks

    Attributes
    ----------
    model: keras.model
        The keras model
    history: keras.callbacks.callbacks.History
        The train history of the model
    """
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

    def saveModel(self, name="", savePath="DataStore/SavedModels/Forecasters/",
                  modelConfig=None):
        """Saves the model into the specified path.

        This function writes the model and some additional details into
        the specified location. The directory naming convention is as
        follows,

        directoryName = yyyy-mm-dd@HH:MM

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

        saveDateTime = str(datetime.datetime.now())[:-10].replace(" ", "@")
        if saveDateTime in os.listdir(savePath):
            message = "model already exists for this datetime"
            raise Exception(message)
        savePath = "{}/{}/".format(savePath, saveDateTime)
        os.mkdir(savePath)
        with open(savePath + "modelSummary.txt", "w+") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        if modelConfig:
            tmp = json.loads(self.model.to_json())
            tmp["name"] = modelConfig["modelName"]
            tmp["trainConfig"] = {}
            tmp["trainConfig"] = modelConfig

        with open(savePath + "modelConfig.json", "w+") as f:
            f.write(self.model.to_json())
        with open(savePath + "history.pickle", "wb+") as f:
            pickle.dump(self.history, f)
        self.model.save(savePath + "model", save_format="tf")

    def loadModel(self, savePath="DataStore/SavedModels/Forecasters/",
                  date=False):
        """Loads the specified model.

        This method is just for preparing the input and exception
        handling.

        Parameters
        ----------
        savePath: str, optional
            The location from which the model is to loaded
        date: str, optional
            Used to specify a certain date. If none, loads the latest. This has
            to be the folder name of the saved model.
        """
        savePath = self.getProjectRoot() + savePath
        modelName = self.__class__.__name__
        savePath += modelName
        if not date:
            if len(os.listdir(savePath)) == 0:
                message = "no saved models present"
                raise Exception(message)
            latestSave = sorted(os.listdir(savePath),
                                key=lambda x: self.getDatetime(x))[-1]
            savePath = "{}/{}".format(savePath, latestSave)
            self.loadAll(savePath)
        elif isinstance(date, str):
            allSaves = os.listdir(savePath)
            if date not in allSaves:
                # Raising exception as directory not present in savePath
                message = "{} not in specified location {}".format(date,
                                                                   savePath)
                raise Exception(message)
            else:
                self.loadAll("{}/{}".format(savePath, date))

    def loadAll(self, path):
        """Loads the specified model

        This method loads the model and other attributes when the
        correct path is passed.

        Parameters
        ----------
        path: str
            The path of the model
        """
        self.model = keras.models.load_model(path+"/model")
        with open(path + "/modelConfig.json") as f:
            config = json.load(f)
        firstLayerConfig = config['config']['layers'][0]['config']
        lastLayerConfig = config['config']['layers'][-1]['config']
        self.lookBack = firstLayerConfig['batch_input_shape'][-1]
        self.forecast = lastLayerConfig['units']

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

    def makePredictions(self, data, batchSize=1):
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
