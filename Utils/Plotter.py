from plotly.utils import PlotlyJSONEncoder as encoder
import plotly.graph_objs as go
import json


def plotModelPrediction(ticker, data, prediction, targetFeature):
    allDataPlot = go.Scatter(x=data.index, y=data[targetFeature])
    figure = go.Figure(data=allDataPlot)
    figure.update_layout(autosize=False,
                         height=700,
                         width=1100)
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
    return json.dumps(figure, cls=encoder)


def testPlot():
    import pandas as pd
    ticker = "IOC"
    df = pd.read_csv("DataStore/StockData/{}.csv".format(ticker))
    scat = go.Scatter(x=df["Date"], y=df["Close"])
    fig = go.Figure(data=scat)
    fig.update_xaxes(
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
    with open("test", "w+") as f:
        f.write(json.dumps(fig, cls=encoder))


if __name__ == "__main__":
    testPlot()
