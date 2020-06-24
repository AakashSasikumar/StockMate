import pandas as pd
import math
import os
from DataStore.APIInterface import YFinance


class DataProcessor():
    """The Data Processing Framework.

    The Data Processing Framework is used to have a unified way to specify the
    data processing methods for different models. This helps any model built to
    be more portable and reusable anywhere.

    Attributes
    ----------
    tickers: list
        A list of all the tickers to be used for the model
    features: list
        A list of features needed for each ticker
    tickerData: dict
        A dictionary containing the pandas DataFrame for each ticker
    interval: str
        Parameter specifying the data interval to be fetched and trained on
    apiSource: DataStore.APIInterface
        An object of the API interface. The source from where the data is to
        be download if not present already

    """

    def __init__(self, tickers, features, interval):
        self.tickers = tickers
        self.features = features
        self.interval = interval.upper()
        self.initFeatures()
        self.loadTickerData()
        self.apiSource = YFinance()

    def initFeatures(self):
        # TODO: Increase number of features
        self.OHLCV = ["open", "high", "low", "close", "volume"]
        self.additionalFeatures = []
        self.allFeatures = []
        self.allFeatures.extend(self.OHLCV)
        self.allFeatures.extend(self.additionalFeatures)

    def loadTickerData(self):
        """Method to load the ticker data

        Loads all the tickers specified in self.tickers, and creates a
        class variable self.tickerData that contains all the ticker data
        """
        self.tickerData = {}
        for ticker in self.tickers:
            path = "DataStore/StockData/{}/{}.csv".format(self.interval,
                                                          ticker)
            if not os.path.isfile():
                self.downloadData(ticker)
            data = pd.read_csv(path, index_col="Date", parse_dates=["Date"])

            data = data.sort_index(ascending=True)
            data = self.getFeatures(data)
            self.tickerData[ticker] = data

    def downloadData(self, ticker):
        if "D" in self.interval:
            self.apiSource.saveIntraDay(ticker)
        elif "M" in self.interval:
            self.apiSource.saveInterDay(ticker, self.interval.lower())

    def getTickerData(self):
        """Returns the ticker data

        Returns
        -------
        tickerData: dict
            The dictionary containing all the ticker data
        """
        return self.tickerData

    def getFeatures(self, data):
        """Method to calculate features

        This method calculates/retrieves the features mentioned in
        self.features and adds their respective columns in the pandas
        DataFrame.

        Returns:
        data: pd.DataFrame
            The dataframe containing all the specified features
        """
        newDF = pd.DataFrame(index=data.index)
        for feature in self.features:
            if feature.lower() not in self.allFeatures:
                raise Exception("{} feature is not available".format(feature))
            if feature.lower() in self.OHLCV:
                newDF[feature] = data[self.getColumnName(data, feature)]
            # TODO: Implement ways to get other features using ta-lib

        return newDF

    def getColumnName(self, data, feature):
        """Method to get proper column name from dataframe

        This method is to make the feature names case insensitive.
        If the Open feature were specified as OpEn, this method would match
        the correct column name from the dataframe and return it. As long as
        the spelling is the same, capitalization doesn't matter.

        Parameters
        ----------
        data: pandas.DataFrame
            The raw ticker data
        feature: str
            The user specified feature

        Returns
        -------
        column: str
            The proper column name of the dataframe

        """
        for column in data.columns:
            if column.lower() == feature.lower():
                return column
        raise Exception("{} feature is not available".format(feature))

    def inputProcessor(self, data, context):
        """Method to process raw ticker data into model-compatible format

        This method must be overrode by a child class, as inputProcessing
        differs for each model.

        Parameters
        ----------
        data: dict
            The dict containing raw ticker data
        context: dict
            A dictionary containing information about the usage of this
            function.
            This dictionary can contain the following keys
                1. "ticker": The tickers for which the raw data has been
                              provided in data
                2. "isTrain": A bool indicating whether this call is for
                              training or testing, as training requires
                              the input as well as the target, whereas for
                              testing only the input is needed

        Returns
        -------
        X: np.array
            The input data for the model
        Y: np.array, optional
            The target for the model (for training). This is required
            if context["isTrain"] is True
        """
        raise NotImplementedError("Must override inputProcessor")

    def outputProcessor(self, modelOut, context):
        """Method to process model output into required values

        This method must be overrode by a child process as outputProcessing
        is different for each model.

        This method is not used for the training process, as it is expected
        that target is already in the required format. For example, assume a
        model is trained on the normalized data for tickers. In such a case,
        to view the data on a graph, the model output needs to be
        de-normalized. In other words, this method is used only during the
        inference phase of the model.

        Parameters
        ----------
        modelOut: np.array
            The raw output from the model
        context: dict
            A dictionary containing information about the usage of this
            function

        """
        raise NotImplementedError("Must override outputProcessor")

    def getTrainingData(self, validationSplit=0.7, shuffle=True,
                        batchSize=64):
        """Method specifying how to prepare training data

        Parameters
        ----------
        validationSplit: float
            The float indicating how much of the data should be used for
            training. The other portion is used for validation.
        shuffle: boolean
            A boolean indicating whether the data should be shuffled before
            training
        batchSize: int
            A number indicating the batchsize of the data for training

        Returns
        -------
        trainDS: tf.data.Dataset
            The input for the model
        validDS: tf.data.Dataset
            The target for the model
        """

        context = {}
        context["isTrain"] = True
        lenTrain = 0
        lenValid = 0
        trainDS = None
        validDS = None
        for i, ticker in enumerate(self.tickers):
            context["ticker"] = ticker
            data = self.tickerData[ticker].copy()
            splitIndex = math.floor(validationSplit * len(data))
            lenTrain += len(data[:splitIndex])
            lenValid += len(data[splitIndex:])
            if i == 0:
                trainDS = self.inputProcessor(data[:splitIndex], context)
                validDS = self.inputProcessor(data[splitIndex:], context)
            else:
                tmpTrain = self.inputProcessor(data[:splitIndex], context)
                tmpValid = self.inputProcessor(data[splitIndex:], context)
                trainDS.concatenate(tmpTrain)
                validDS.concatenate(tmpValid)
        if shuffle:
            trainDS.shuffle(lenTrain)
            validDS.shuffle(lenValid)
        trainDS = trainDS.batch(batchSize).prefetch(1)
        validDS = validDS.batch(batchSize).prefetch(1)
        return trainDS, validDS
