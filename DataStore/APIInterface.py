import yfinance as yf
from lxml.html import fromstring
import random
import requests
import os
import pandas as pd


class YFinance():
    """A wrapper for the Yahoo Finance API

    This is actually a wrapper over a wrapper. This was done to implement
    some more specific behavior over the original.

    Attributes
    ----------
    autoRotate: bool
        A feature that allows you to break the barrier of api limits.
        When provided a list of apiKeys, it will automatically rotate
        them along with different proxies to get data
    suffix: str
        A string indicating which exchange we want to get the stock data from
    intraDayIntervals: list
        A list of intervals supported by this API
    proxyList: list
        A list of proxy addresses
    """
    def __init__(self, autoRotate=False, exchange="NSE"):
        self.autoRotate = autoRotate

        if exchange == "NSE":
            self.suffix = ".NS"
        else:
            self.suffix = ""

        if self.autoRotate:
            self.proxyList = self.getProxies()

        self.intraDayIntervals = ["1m", "2m", "5m", "15m", "30m", "1h"]

    def getIntraDay(self, ticker, start=None):
        """Method to get the intra-day data for a stock

        Parameters
        ----------
        ticker: str
            The name of the stock
        start: str, optional
            A date indicating from which day we need the data. If None,
            returns the entire historical data

        Returns
        -------
        data: pandas.DataFrame
            The dataframe object of the raw ticker data
        """
        tickerSymbol = ticker+self.suffix
        tickerObj = yf.Ticker(tickerSymbol)
        interval = "1d"

        if start is None:
            period = "max"
            args = {"interval": interval, "period": period}
            return self.getData(tickerObj, args)
        else:
            start = start
            args = {"interval": interval, "start": start}
            return self.getData(tickerObj, args)

    def getInterDay(self, ticker, interval):
        """Method to get the inter-day data for a stock

        Parameters
        ----------
        ticker: str
            The name of the stock
        interval: str
            The interval width of the data. The list of supported
            intervals are in the self.intraDayIntervals object.

        Returns
        -------
        data: pandas.DataFrame
            The dataframe object of the raw ticker data
        """
        tickerSymbol = ticker + self.suffix
        tickerObj = yf.Ticker(tickerSymbol)

        if interval == "1m":
            period = "7d"
        elif interval != "1m" and interval in self.intraDayIntervals:
            period = "60d"
        args = {"period": period, "interval": interval}
        return self.getData(tickerObj, args)[:-1]

    def getData(self, tickerObj, arguments):
        """Method to get the raw ticker data

        Parameters
        ----------
        tickerObj: yfinance.Ticker
            The ticker object of the yfinance module
        arguments: dict
            A dictionary containing the parameters for getting the raw ticker
            data

        Returns
        -------
        data: pandas.DataFrame
            The dataframe of the raw ticker data
        """
        if self.autoRotate:
            proxy = random.choice(self.proxyList)
        else:
            proxy = None
            return tickerObj.history(proxy=proxy, **arguments)
        while True:
            try:
                data = tickerObj.history(proxy=proxy, **arguments)
                break
            except Exception as e:
                print(e)
                self.proxyList.remove(proxy)
                proxy = random.choice(self.proxyList)
        return data

    def saveIntraDay(self, ticker, start=None, savePath="DataStore/StockData"):
        """Method to save the ticker data to a specified location

        Parameters
        ----------
        ticker: str
            The name of the stock
        start: str, optional
            A date indicating from which day we need the data. If None,
            returns the entire historical data
        """
        data = self.getIntraDay(ticker, start=None)
        if not os.path.isdir(savePath+"/1D"):
            os.mkdir(savePath+"/1D")
        savePath = savePath + "/1D/{}.csv".format(ticker)
        data.to_csv(savePath)

    def saveInterDay(self, ticker, interval, savePath="DataStore/StockData"):
        """Method to save the interday data to a specified location

        Parameters
        ----------
        ticker: str
            The name of the stock
        interval: str
            The interval width of the data. The list of supported
            intervals are in the self.intraDayIntervals object.

        """
        savePath = savePath + "/{}".format(interval).upper()
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        data = self.getInterDay(ticker, interval)
        data.index.names = ["Date"]
        savePath = savePath + "/{}.csv".format(ticker)
        if os.path.isfile(savePath):
            # append to existing file
            oldData = pd.read_csv(savePath, index_col="Date",
                                  parse_dates=["Date"])
            oldData = oldData.sort_index(ascending=True)
            data = data[data.index > oldData.index[-1]]
            data = pd.concat([oldData, data])

        data.to_csv(savePath)

    def getProxies(self, num=20):
        """Method to scrape/load the list of valid proxies

        Parameters
        ----------
        num: int
            The number of proxy addresses to scrape

        Returns
        -------
        proxyList: list
            A list of all proxy addresses
        """
        proxyList = []
        # if "proxies.txt" in os.listdir():
        #     with open("proxies.txt") as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             proxyList.append(line.strip())
        #     return proxyList
        url = 'https://free-proxy-list.net/'
        response = requests.get(url)
        parser = fromstring(response.text)
        for i in parser.xpath('//tbody/tr')[:num]:
            proxy = ":".join([i.xpath('.//td[1]/text()')[0],
                              i.xpath('.//td[2]/text()')[0]])
            proxyList.append(proxy)
        return proxyList
