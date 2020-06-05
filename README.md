# StockMate

A python based tool to build agents and models for stock price forecasting and trade automation.

`StockMate` is just a temporary name, will change it once I come up with a better one.

## Table of Contents

1. [Getting Started](#Getting-Started)
    - [Prerequisites](#Prerequisites)
2. [What is StockMate?](#What-is-StockMate?)
    - [Terminology](#Terminology)
    - [Features](#Features)
    - [Planned Features](#Planned-Features)
3. [Usage](#Usage)
    - [Setup](#Setup)
    - [Saving Stock Data](#Saving-Stock-Data)
    - [Forecaster Creation](#Forecaster-Creation)

## Getting Started

### Prerequisites

1. plotly dash
2. selenium
3. phantomjs driver
4. tensorflow 2.0 or greater
    - As of June 5th 2020, tensorflow 2.0 has an error when loading a saved model containing LSTM layers. So the workaround for this is to install tf-nightly as they have patched this in this version.
5. lxml

## What is StockMate

StockMate is a Python based tool where you can create models for stock price prediction and trade automation. Using StockMate you can use the provided APIs to get the latest stock data, build and test models for forecasting, etc. Currently StockMate's data retrieval is built around India's National Stock Exchange (NSE), and support for other exchanges isn't built yet.

### Terminology

#### 1. Forecasters

Forecasters are a regression models for predicting stock prices. Forecasters can be trained for individual stocks or for entire indices. Basic models have been implemented already and can be used out of the box. There are 3 main parameters for forecasters ie,

1. `Stock Data` - Some of the models support multivariate data, to make sure, check the documentation for each model
2. `forecast` - The number of days in the future for which prices are to be predicted
3. `lookBack` - The number of days to be used to make `forecast` predictions.

Using this forecast information, we can make decisions on whether to buy to sell stocks.

#### 2. Agents

Agents are used to automate trading. Agents decide when to buy, sell or hold stock. Currently there are no free trading apis, so the next best solution for automated trading is to make a chatbot that tells you when to buy and sell.

#### 3. Data Processing Framework (DPF)

The Data Processing Framework was something that had to be build to make the created models more portable. Think of the model as a black box; Each model will have its own way of handling,

1. The incoming raw stock data
    - Maybe this raw data is normalized, or some other operations are applied to it before being fed to the model
2. The model output
    - The the input was normalized for a regressor, the model output would have to be de-normalized to get back the predicted prices

These two functionalities have been abstracted out of the model and incorporated into a `Data Processing Framework (DPF)`. Basically what this means is that, to create a model you would have define your own DPF by inheriting `Core.DataProcessor` and overriding the following methods,

1. `inputProcessor()`
2. `outputProcessor()`
3. `getTrainingData()`

Please check the documentation under `Core.DataProcessor` to see what the parameters that are passed into it and what the expected outputs are.

Also, two DPFs have already been implemented for handing univariate and multivariate stock data respectively. You can find them under `Examples/Processors/BasicProcessors.py`

### Features

Currently, the implemented features for StockMate include

1. A framework for regression models
2. A web UI for forecaster creation
    - I have no skills in making good UIs or websites; I shamelessly copied [an open source dashboard](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2).
3. A framework for up-to-date stock data retrieval
4. A tool for updating NSE Indices

The following models have been implemented,

#### Forecasters

1. ANN.BasicRegressor
2. ANN.DenseRegressor
3. LSTM.BasicLSTM
4. LSTM.DenseLSTM
5. CNN.BasicCNN
6. CNN.WaveNet

#### Agents

1. TBA

### Planned Features

1. Packaging this repo
2. Agent building framework
3. Fully fledged website to view and modify forecasters and agents
4. More customization for forecasters in the UI
5. Semi-trade-automation by means of a telegram chatbot

## Usage

### Setup

#### 1. API Key(s)

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

### Saving Stock Data

#### 1. Data retrieval for a single stock (TCS)

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

#### 2. Data retrieval for an entire index

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

### Forecaster Creation

```python
from DataStore import Indices
from Examples.Processors import MultiVarProcessor
from Models.Forecasters.CNN import WaveNet

indices = Indices.NSEIndices().getIndices()
category = "Broad Market Indices"
index = "NIFTY 50"

constituents = indices["type"][category][index]

# how many days to look back to make prediction
lookBack = 30
# how many days in future to predict
forecast = 5

# target feature is the feature that we want to predict
# in this case it is the closing price
dpf = MultiVarProcessor(tickers=constituents, features=["open", "high", "low", "close", "volume"],
lookBack=lookBack, forecast=forecast, targetFeature="close", isSeq2Seq=True)

model = WaveNet()
model.assignDataProcessor(dpf)
model.buildModel(learningRate=1e-5)
model.train(validationSplit=0.9, epochs=1000, batchSize=64)
# By default, models will save in DataStore/SavedModels/
model.saveModel("waveNetTest")
# saved models can be loaded again by calling model.loadModel("name") and trained/inferenced upon
```
