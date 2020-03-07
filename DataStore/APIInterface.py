import requests


class AlphaVantage():

    def __init__(self, apiKey):
        self.apiKey = apiKey

    def buildQuery(self, function, ticker, interval=None, size="full",
                   exchange="NSE", datatype="csv"):

        ticker = "{}:{}".format(exchange, ticker)

        self.alphaVantageURLBase = "https://www.alphavantage.co"

        if interval:
            intervalStr = "&interval={}min".format(interval)
        else:
            intervalStr = ""

        template = ("query?function={function}&symbol={ticker}"
                    "{interval}&apikey={apiKey}&"
                    "outputsize={outputsize}&datatype={datatype}")

        query = template.format(function=function, ticker=ticker,
                                interval=intervalStr, apiKey=self.apiKey,
                                outputsize=size, datatype=datatype)

        return "{}/{}".format(self.alphaVantageURLBase, query)

    def retreiveIntraDay(self, ticker, interval=1, datatype="csv"):

        function = "TIME_SERIES_INTRADAY"

        alphaVantageURL = self.buildQuery(function, ticker,
                                          interval, datatype=datatype)

        return self.getResponse(alphaVantageURL)

    def retreiveDailyAdjusted(self, ticker, datatype="csv"):

        function = "TIME_SERIES_DAILY_ADJUSTED"

        alphaVantageURL = self.buildQuery(function, ticker,
                                          datatype=datatype)

        return self.getResponse(alphaVantageURL)

    def getResponse(self, url):

        try:
            response = requests.get(url)
        except Exception as e:
            print(e.__traceback__)

        if response:
            return response.content.decode("utf8")
