import json

import requests

from CONFIG import ALPHA_VANTAGE_API


def buildQuery(function, ticker, interval, compact="full",
               apikey=ALPHA_VANTAGE_API, exchange="NSE"):

    ticker = "{}:{}".format(exchange, ticker)

    alphaVantageURLBase = "https://www.alphavantage.co/"
    template = ("query?function={function}&symbol={ticker}"
                "&interval={interval}min&apikey={apiKey}&"
                "compact={compact}")

    query = template.format(function=function, ticker=ticker,
                            interval=interval, apiKey=apikey,
                            compact=compact)

    return "{}{}".format(alphaVantageURLBase, query)


def retreiveIntraDay(ticker, interval=1):
    function = "TIME_SERIES_INTRADAY"

    alphaVantageURL = buildQuery(function, ticker, interval)
    print(alphaVantageURL)

    try:
        response = requests.get(alphaVantageURL)
    except Exception as e:
        print(e.__traceback__)

    if response:
        return json.loads(response.content)


def main():
    data = retreiveIntraDay("ABFRL")
    print(data)


if __name__ == "__main__":
    main()
