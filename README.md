# StockMate

A python based tool to build agents and models for stock price prediction and trade automation.

## Requirements

1. plotly dash
2. selenium
3. phantomjs driver
4. tensorflow 2.0 or greater
5. lxml

## Features

1. Building Forecasters
2. Viewing forecaster predictions
3. Automatic stock data retrieval from AlphaVantage
4. Automated NSE Indices and constituents list retrieval

## Planned Features

1. Building Agents
2. Fully fledged website to view and modify forecasters and agents
3. More customization for forecasters
4. Semi-trade-automation by means of a telegram chatbot

## Methods of Stock Price Predictions

### 1. Forecasters

Forecasters are a regression based way of predicting stock prices. Forecasters can be trained for individual stocks or for entire indices. Basic models have been implemented already and can be used out of the box. There are 3 main parameters for forecasters ie,

1. `Stock Data` - Some of the models support multivariate data, to make sure, check the documentation for each model
2. `forecast` - The number of days in the future for which prices are to be predicted
3. `lookBack` - The number of days to be used to make `forecast` predictions.

Using this forecast information, we can make decisions on whether to buy to sell stocks.

### 2. Agents

Agents are used to automate trading completely. Agents decide when to buy, sell or hold stock. Currently there are no free trading apis, so the next best solution for automated trading is to make a chatbot that tells you when to buy and sell.

## Setup Instructions

### Stock Data Retrieval

1. Create an api key on [AlphaVantage](https://www.alphavantage.co/support/#api-key)
    - OPTIONAL: Create multiple additional api keys using alternate email addresses and temporary email generators. If multiple API keys are present, they can be rotated when retreiving large amounts of stock data as there are daily limits for each key.
2. Create a file named `CONFIG.py` in the root location of the repository
3. Create a variable for the api key, ie

    ```python
    ALPHA_VANTAGE_API_KEY = "your api key here"  # with the quotes
    ```

    - If additional api keys were made, make separate variables for each one and create a final variable as a list of all these api keys, ie

    ```python
    KEY1 = "apiKey1"
    KEY2 = "apiKey2"
    KEY3 = "apiKey3"
    KEY_LIST = [KEY1, KEY2, KEY3]
    ```

4. Run `setup.py` to scrape and save the latest NSEIndices and its constituents

## Usage

### Saving Stock Data

1. Data retrieval for a single stock (TCS)

    ```python
    from DataStore.APIInterface import AlphaVantage
    import CONFIG

    ticker = "TCS"

    source = AlphaVantage(CONFIG.KEY1)

    data = source.getDailyAdjusted(ticker)
    # The data returned is a csv as a single string
    # In this case, we will create and write this data into a csv file.
    with open("DataStore/StockData/{}".format(ticker), "w+") as f:
        f.write(data)

    ```

2. Data retrieval for an entire index

    ```python
    from DataStore import Indices
    from DataStore.APIInterface import AlphaVantage
    import CONFIG

    # For category and index names, check the file saved by running the setup.py file
    category = "Broad Market Indices :"
    index = "NIFTY 50"

    nse = Indices.NSEIndices()
    indices = nse.getIndices()

    # for a single api key
    source = AlphaVantage(CONFIG.KEY1)

    # for multiple api keys
    source = AlphaVantage(CONFIG.KEY_LIST, autoRotate=True)

    constituents = indices["type"][category][index]

    for stock in constituents:
        with open("{}.csv".format(stock), "w+") as f:
            f.write(source.getDailyAdjusted(stock))
    ```

    - AutoRotate is a feature that takes a list of api keys and rotates them so that the daily limit can be breached. It also scrapes a list of proxy addresses so that AlphaVantage doesn't block the source IP.
