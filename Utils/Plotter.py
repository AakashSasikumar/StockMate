from plotly.utils import PlotlyJSONEncoder as encoder
import plotly.graph_objs as go
import json


def getModelPredictionFigure(ticker, data, prediction, targetFeature,
                             plotType):
    if plotType == "lineChart":
        plot = makeLineChart(data, targetFeature)
    else:
        # Type not supported
        raise Exception("Plot type {} not supported".format(plotType))

    figure = go.Figure(data=plot)
    figure.update_layout(autosize=False,
                         height=700,
                         width=1100,
                         title=ticker)
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


def makeLineChart(data, targetFeature):
    plot = go.Scatter(x=data.index, y=data[targetFeature])
    return plot
