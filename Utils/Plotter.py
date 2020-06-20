from plotly.utils import PlotlyJSONEncoder as encoder
import plotly.graph_objs as go
import json
from datetime import timedelta
import pandas as pd


def getForecasterPredictionFigure(ticker, data, prediction, targetFeature,
                                  plotType):
    """Method to plot stock data and model predictions

    This method uses plotly to plot the data and returns the encoded json
    format to the UI.

    Parameters
    ----------
    ticker: str
        The ticker symbol which is to be plotted. This is used as the title
        of the plot
    data: pandas.DataFrame
        The raw data of the ticker
    prediction: numpy.array
        The model's output after being passed through
        DataProcessor.outputProcessor()
    targetFeature: str
        The target feature for which the model was trained
    plotType: str
        The type of plot that is to be plotted. Currently these plots are
        supported,
            1. Line Chart

    Returns
    -------
    figure: str
        The plotly figure encoded into a JSON format
    """
    if plotType == "lineChart":
        basePlot = makeLineChart(data.index.date, data[targetFeature],
                                 name="Actual")
    else:
        # Type not supported
        raise Exception("Plot type {} not supported".format(plotType))
    predictionPlots = plotForecasterPrediction(data, prediction)
    figure = go.Figure()
    figure.add_trace(basePlot)
    for plots in predictionPlots:
        figure.add_trace(plots)
    figure.update_layout(autosize=False,
                         height=700,
                         width=1100,
                         title=ticker,
                         showlegend=False,
                         dragmode="pan")

    figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    figure = json.dumps(figure, cls=encoder)
    return figure


def getAgentPredictionFigure(ticker, data, prediction, plotType):
    """Method to plot stock data and model predictions

    This method uses plotly to plot the data and returns the encoded json
    format to the UI.

    Parameters
    ----------
    ticker: str
        The ticker symbol which is to be plotted. This is used as the title
        of the plot
    data: pandas.DataFrame
        The raw data of the ticker
    prediction: numpy.array
        The model's output after being passed through
        DataProcessor.outputProcessor()
    plotType: str
        The type of plot that is to be plotted. Currently these plots are
        supported,
            1. Line Chart

    Returns
    -------
    figure: str
        The plotly figure encoded into a JSON format
    """
    if plotType == "lineChart":
        basePlot = makeLineChart(data.index.date, data["Close"],
                                 name="Actual")
    else:
        # Type not supported
        raise Exception("Plot type {} not supported".format(plotType))

    predictionPlot = plotAgentPrediction(data, prediction)
    figure = go.Figure()
    figure.add_trace(basePlot)
    figure.add_trace(predictionPlot)
    figure.update_layout(autosize=False,
                         height=700,
                         width=1100,
                         title=ticker,
                         showlegend=False,
                         dragmode="pan")

    figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    figure = json.dumps(figure, cls=encoder)
    return figure


def makeLineChart(x, y, line=None, name=None):
    """Method to plot a line chart

    Parameters
    ----------
    x: numpy.array
        The values for X axis
    y: numpy.array
        The values for y axis
    line: dict
        The line format parameters
    name: str
        The name for the line
    """
    plot = go.Scatter(x=x, y=y, line=line, name=name)
    return plot


def plotForecasterPrediction(data, predictions):
    """Method to plot all the predictions

    This method takes care of preserving the dates of the predictions.

    Parameters
    ----------
    data: pandas.DataFrame
        The raw ticker data
    predictions: numpy.array
        The model's output after being passed through
        DataProcessor.outputProcessor()

    Returns
    -------
    allPredictions: list
        A list of all the plotly plot objects
    """
    predictionFormat = {"color": "firebrick",
                        "dash": "dot"}
    startDate = data.index[-1] + timedelta(days=1)
    forecast = len(predictions[0])
    mappedDates = zip(data.index[::-forecast], predictions[::-1])
    allPredictionPlots = []
    for i, zippedData in enumerate(mappedDates):
        date, prediction = zippedData
        if i == 0:
            startDate = date + timedelta(days=1)
            endDate = date + timedelta(days=5)
            dateRange = pd.date_range(start=startDate, end=endDate)
            plot = makeLineChart(dateRange.date, prediction,
                                 line=predictionFormat, name="Predicted")
        else:
            startIndex = list(data.index).index(date) + 1
            endIndex = list(data.index).index(date) + 6
            dateRange = data.index[startIndex:endIndex]
            plot = makeLineChart(dateRange.date, prediction,
                                 line=predictionFormat, name="Predicted")

        allPredictionPlots.append(plot)
    return allPredictionPlots
