from tensorflow import keras
import os
import pickle
import dill
import datetime
import json


class RegressorBase():
    """A wrapper for all keras based regression networks

    Attributes
    ----------
    model: keras.model
        The keras model
    history: keras.callbacks.callbacks.History
        The train history of the model
    dataProcessor: Core.DataProcessor
        The data processor for the model
    """
    def __init__(self):
        self.history = None

    def buildModel(self, learningRate=None):
        """Builds the model and sets the class attribute

        Parameters
        ----------
        learningRate: float, optional
            The learning rate for the optimizer
        """
        raise NotImplementedError("Must override buildModel")

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

    def saveModel(self, name,
                  savePath="DataStore/SavedModels/Forecasters/"):
        """Saves the model into the specified path.

        NOTE: If name already exists in the directory, calling saveModel
        will overwrite the existing saved files.

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
        4. model
            The trained model
        5. dataProcessor
            The data processor used for this model

        Parameters
        ----------
        name: str
            An optional personal name given to the model
        savePath: str, optional
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

        modelPath = "{}/{}/".format(savePath, name)
        if name not in os.listdir(savePath):
            os.mkdir(modelPath)

        savePath = modelPath
        with open(savePath + "modelSummary.txt", "w+") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        with open(savePath + "modelConfig.json", "w+") as f:
            f.write(self.model.to_json())
        if self.history:
            with open(savePath + "history.pickle", "wb+") as f:
                pickle.dump(self.history, f)
        self.model.save(savePath + "model", save_format="tf",
                        include_optimizer=True)
        with open(savePath + "dataProcessor.dill", "wb+") as f:
            dill.dump(self.dataProcessor, f)
        with open(savePath + "ForecasterInfo.json", "w+") as f:
            writeJson = {}
            writeJson["tickers"] = self.dataProcessor.tickers
            writeJson["features"] = self.dataProcessor.features
            writeJson["savedTime"] = str(datetime.datetime.now())[:-10]
            json.dump(writeJson, f)

    def loadModel(self, name, savePath="DataStore/SavedModels/Forecasters/"):
        """Loads the specified model.

        This method is just for preparing the input and exception
        handling.

        Parameters
        ----------
        savePath: str, optional
            The location from which the model is to loaded
        name: str
            Used to specify a certain date. If none, loads the latest. This has
            to be the folder name of the saved model.
        """
        savePath = self.getProjectRoot() + savePath
        modelName = self.__class__.__name__
        modelPath = savePath + modelName
        if name not in os.listdir(modelPath):
            message = "Could not find {} in {}".format(name, savePath)
            raise Exception(message)
        savePath = modelPath
        savePath = "{}/{}".format(savePath, name)
        model, dp = RegressorBase.loadAll(savePath)
        self.model = model
        self.dataProcessor = dp

    def loadAll(path):
        """Loads the specified model

        This method loads the model and other attributes when the
        correct path is passed.

        Parameters
        ----------
        path: str
            The path of the model
        """
        model = keras.models.load_model(path+"/model")
        with open(path + "/dataProcessor.dill", "rb") as f:
            dataProcessor = dill.load(f)
        return model, dataProcessor

    def assignDataProcessor(self, dataProcessor):
        """Method to assign the dataProcessor

        Parameters
        ----------
        dataProcessor: Object
            The child of Core.DataProcessor with all methods
            implemented
        """
        self.dataProcessor = dataProcessor
        self.lookBack = self.dataProcessor.lookBack
        self.forecast = self.dataProcessor.forecast

    def train(self, epochs=1000, earlyStopping=True,
              patience=15, callbacks=[], shuffle=False,
              batchSize=64, validationSplit=0.7):
        """The method to start training the model

        Parameters
        ----------
        validationSplit: float
            The ratio in which the input data has to be split into
            training and validation.
        epochs: int, optional
            The number of epochs to train for
        earlyStopping: boolean, optional
            Specify whether early stopping based on validation
            loss is required (recommended to keep it as True)
        patience: int, optional
            The number of epochs to wait for early stopping
        callbacks: list
            custom callbacks may be specified for training
        validationSplit: float
            The float indicating how much of the data should be used for
            training. The other portion is used for validation.
        shuffle: boolean
            A boolean indicating whether the data should be shuffled before
            training
        batchSize: int
            A number indicating the batchsize of the data for training
        """
        if self.dataProcessor is None:
            message = "DataProcessor not specified for this model"
            raise Exception(message)

        keras.backend.clear_session()
        if earlyStopping:
            callback = keras.callbacks.EarlyStopping(patience=patience)
            callbacks.append(callback)

        trainDS, validDS = self.dataProcessor.getTrainingData(validationSplit,
                                                              shuffle,
                                                              batchSize)

        history = self.model.fit(trainDS, epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=validDS)
        self.history = history.history

    def makePredictions(self, data, context):
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
        procInput = self.dataProcessor.inputProcessor(data, context)
        prediction = self.model.predict(procInput)
        return self.dataProcessor.outputProcessor(prediction, context)
