import pandas as pd


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

    """

    def __init__(self, tickers, features):
        self.tickers = tickers
        self.features = features

        self.loadTickerData()

    def loadTickerData(self):
        """Method to load the ticker data

        Loads all the tickers specified in self.tickers, and creates a
        class variable self.tickerData that contains all the ticker data
        """
        self.tickerData = {}
        for ticker in self.tickers:
            data = pd.read_csv("DataStore/StockData/{}.csv".format(ticker),
                               index_col="timestamp")
            data = data.sort_index()
            data = self.getFeatures(data)
            self.tickerData[ticker] = data

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
        # TODO: Implement ways to get other features
        return data

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
                1. "tickers": The tickers for which the raw data has been
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
        validationData: tuple,   optional
            A tuple of validX, validY. This is optional even if
            context["isTrain"] is True. Providing validationData
            is recommended as it supports use of callbacks.
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

    def processInput(self, train=True):
        context = {}
        context["isTrain"] = train
        test, train = self.inputProcessor(self.tickerData[self.tickers[0]],
                                          context)
