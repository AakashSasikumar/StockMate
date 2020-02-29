import requests

from CONFIG import ALPHA_VANTAGE_API


def buildQuery(function, ticker, interval, size="full",
               apikey=ALPHA_VANTAGE_API, exchange="NSE",
               datatype="csv"):

    ticker = "{}:{}".format(exchange, ticker)

    alphaVantageURLBase = "https://www.alphavantage.co/"
    template = ("query?function={function}&symbol={ticker}"
                "&interval={interval}min&apikey={apiKey}&"
                "outputsize={outputsize}&datatype={datatype}")

    query = template.format(function=function, ticker=ticker,
                            interval=interval, apiKey=apikey,
                            outputsize=size, datatype=datatype)

    return "{}{}".format(alphaVantageURLBase, query)


def retreiveIntraDay(ticker, interval=1, datatype="csv"):
    function = "TIME_SERIES_INTRADAY"

    alphaVantageURL = buildQuery(function, ticker, interval, datatype=datatype)

    try:
        response = requests.get(alphaVantageURL)
    except Exception as e:
        print(e.__traceback__)

    if response:
        return response.content.decode("utf8")


def main():
    data = retreiveIntraDay("ABFRL")
    print(data)


if __name__ == "__main__":
    main()
