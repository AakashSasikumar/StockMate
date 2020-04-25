import requests
import time
import urllib
from Utils import AutoRotate as ar


class AlphaVantage():
    """A wrapper for the AlphaVantage stock data API

    Attributes
    ----------
    apiKey: str or list
        The API key for the AlphaVantage API, a list of api keys have to
        be given when using autototate
    reqPerMin: int
        The limit of number of requests per minute set by AlphaVantage
    reqPerDay: int
        The day limit of number of requesters set by AlphaVantage
    startTime: time.time()
        The time at which the first request had been sent in a
        60 second bracket
    count: int
        The number of requests made within a 60 second bracket
    autoRotate: bool
        A feature that allows you to break the barrier of api limits.
        When provided a list of apiKeys, it will automatically rotate
        them along with different proxies to get data.

    Note: The autorotate feature is limited to len(apiKeys) * reqPerDay

    """
    def __init__(self, apiKey, reqPerMin=5, reqPerDay=500,
                 autoRotate=False, apiUsageDataLoc="APIKeyUsageData.json"):
        self.apiKey = apiKey
        self.reqPerMin = reqPerMin
        self.reqPerDay = reqPerDay
        self.startTime = None
        self.count = 0
        self.autoRotate = autoRotate

        if autoRotate:
            # Using autorotate but apiKey isn't a list
            if not isinstance(apiKey, list):
                message = "apiKey must be a list when using autoRotate"
                raise Exception(message)
            self.apiKeys = apiKey
            self.apiKey = apiKey[0]
            ar.init(self.apiKeys, apiUsageDataLoc, self.reqPerDay)
        if not autoRotate:
            # Not using autorotate but apiKey isn't a list
            if not isinstance(apiKey, str):
                message = "apiKey must be str when using autoRotate"
                raise Exception(message)

    def buildQuery(self, function, ticker, interval=None, size="full",
                   exchange="NSE", datatype="csv"):
        """Returns final URL for the alphavantage api.

        This function builds query after taking all the parameters as input and
        returns the URL.

        Parameters
        ----------
        function: str
            The type of data to be retrieved
        ticker: str
            The ticker for which the data is to be retrieved
        interval: str
            The interval of the data (daily, hourly, 5 minutes..)
        size: str
            1. "full": returns the entire historical data
            2. "compact": returns a small sample of the data
        exchange: str
            The exchange from which the ticker data is to be retrieved
        datatype:
            1. "json": returns the data as a json
            2. "csv": returns the data as a CSV

        Returns
        -------
        queryURL: str
            The urlencoded form of the query
        """
        ticker = "{}:{}".format(exchange, ticker)

        self.alphaVantageURLBase = "https://www.alphavantage.co"

        if interval:
            intervalStr = "{}min".format(interval)
        else:
            intervalStr = ""

        arguments = {"function": function,
                     "symbol": ticker,
                     "interval": intervalStr,
                     "apikey": self.apiKey,
                     "outputsize": size,
                     "datatype": datatype}

        if not interval:
            del arguments["interval"]

        query = urllib.parse.urlencode(arguments)
        queryURL = "{}/query?{}".format(self.alphaVantageURLBase, query)
        return queryURL

    def getIntraDay(self, ticker, interval=1, datatype="csv"):
        """Returns intraday data

        Parameters
        ----------
        ticker: str
            The stock for which the data is to be retrieved
        interval: str
            The interval of the data (daily, hourly, 5 minutes..)
        datatype:
            1. "json": returns the data as a json
            2. "csv": returns the data as a CSV

        Returns
        -------
        response: str
            The intraday data
        """
        function = "TIME_SERIES_INTRADAY"

        proxy = None
        if self.autoRotate:
            self.apiKey, proxy = ar.getKeyAndProxy()
        alphaVantageURL = self.buildQuery(function, ticker,
                                          interval, datatype=datatype)

        return self.getResponse(alphaVantageURL, proxy)

    def getDailyAdjusted(self, ticker, datatype="csv"):
        """Returns daily adjusted data

        Parameters
        ----------
        ticker: str
            The stock for which the data is to be retrieved
        interval: str
            The interval of the data (daily, hourly, 5 minutes..)
        datatype:
            1. "json": returns the data as a json
            2. "csv": returns the data as a CSV

        Returns
        -------
        response: str
            The intraday data

        """
        function = "TIME_SERIES_DAILY_ADJUSTED"

        proxy = None
        if self.autoRotate:
            self.apiKey, proxy = ar.getKeyAndProxy()
        alphaVantageURL = self.buildQuery(function, ticker,
                                          datatype=datatype)
        response = self.getResponse(alphaVantageURL, proxy)
        return response

    def getResponse(self, url, proxy=None):
        """Returns the response after querying

        Parameters
        ----------
        url: str
            The final API query request

        Returns
        -------
        response: str
            The response from the API
        """
        if not self.startTime:
            self.startTime = time.time()

        limitReached, timeTillNext = self.requestLimitReached()

        if limitReached:
            time.sleep(timeTillNext)
            self.startTime = time.time()
            self.count = 0

        try:
            if proxy:
                proxy = {'http': proxy, 'https:': proxy}
                response = requests.get(url, proxies=proxy)
            else:
                response = requests.get(url)
        except Exception as e:
            print(e)

        self.count += 1

        if response:
            return response.content.decode("utf8")

    def requestLimitReached(self):
        """A simple measure to meet the API request limit

        The AlphaVantage API only allows 5 requests per minute. If this
        limit is breached, it returns an error, so a simple check is
        implemented to make the system wait if a 6th request is to be
        made within 60 seconds.

        Returns:
            limitReached: bool
                True: The limit has been breached, and the system has to wait
                False: The limit has not been breached
            timeTillNext: int
                The number of seconds till a new request can be sent
        """
        currentTime = time.time()
        if ((currentTime - self.startTime) <= 60) and \
           self.count == self.reqPerMin:
            limitReached = True
            timeTillNext = self.startTime + 60 - currentTime
        else:
            limitReached = False
            timeTillNext = 0

        return limitReached, timeTillNext
