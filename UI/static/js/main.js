function submitAPIKey() {
    //TODO:
    // Finish implementing this..
    var apiKey = document.getElementById("apiKey").value;
    var payload = new Object();
    payload["apiKey"] = apiKey;

    sendPayload(payload, "/submitTelegramAPIKey")
}

function resetRoot() {
    sendPayload(new Object(), "/resetTelegramRoot")
}

function forecasterSelected() {
    showForecasterDescription();
}

function showForecasterDescription() {
    var selectedModel = document.getElementById("forecasterSelection").value;
    for (model in allForecasters) { 
        if (allForecasters.hasOwnProperty(model)) {
            modelDescId = model+"ForecasterDescription";
            var element = document.getElementById(modelDescId);
            if (selectedModel == model) {
                element.style.display ="block";
            }
            else {
                element.style.display = "none";
            }
        }
    }
}

function indexCategorySelected(element) {
    category = element.value;
    for (cat in indices) {
        if (indices.hasOwnProperty(cat)) {
            if (cat == category) {
                options = "";
                for (index in indices[cat]) {
                    tmp = "<option value='" + index + "'>" +index + "</option>";
                    options += tmp;
                }
                document.getElementById("indexSelection").style.display = "block";
                document.getElementById("indexName").innerHTML = options
            }
        }
    }
}

function indexNameSelected(element) {
    category = document.getElementById("indexCategory").value;
    indexName = element.value;
    constituents = indices[category][indexName]["constituents"];
    options = "";
    for (var i = 0; i < constituents.length; i++) {
        if (i == 0) {
            tmp = "<option value='choose'>Choose tickers</option>";
            options += tmp;
            tmp = "<option value='all'>All Stocks</option>";
            options += tmp;
        }
        tmp = "<option value='" + constituents[i] + "'>" + constituents[i] + "</option>";
        options += tmp;
    }
    document.getElementById("stockTickerSelection").style.display = "block";
    document.getElementById("stockTickers").innerHTML = options;
    document.getElementById("tickerSelector").style.display = "block";
    document.getElementById("tickerSelectorButtons").style.display = "block";
    if (!madeTagEditor) {
        initializeTagEditor(constituents);
        madeTagEditor = true;
    }
}
var madeTagEditor = false;
function initializeTagEditor(constituents) {
    $("#selectedTickers").tagEditor({
        placeholder: "Stock Tickers .....",
        forceLowercase: false,
        maxTags: 200,
        maxLength: 10
    });
}

function stockNameSelected(element) { 
    category = document.getElementById("indexCategory").value;
    indexName = document.getElementById("indexName").value;
    ticker = element.value;
    constituents = indices[category][indexName]["constituents"];
    if (ticker == "all") {
        for (var i = 0; i < constituents.length; i++) {
            $("#selectedTickers").tagEditor("addTag", constituents[i]);
        }
    }
    else {
        $("#selectedTickers").tagEditor("addTag", ticker);
    }
}

function clearSelectedTickers(element) {
    if (element.id == "tickerClear") {
        var id = "#selectedTickers";
    }
    else if (element.id == "featureClear") {
        var id = "#selectedFeatures";
    }
    var tags = $(id).tagEditor("getTags")[0].tags;
    for (var i = 0; i < tags.length; i++) {
        $(id).tagEditor("removeTag", tags[i]);
    }
}

var featureSelectorMade = false;
function initFeatureSelector() {
    $("#selectedFeatures").tagEditor({
        placeholder: "Select Features .....",
        forceLowercase: false,
      });
}

var targetFeatureShown = false;

function featureSelected(element) {
    if (!featureSelectorMade) {
        initFeatureSelector()
        featureSelectorMade = true;
        document.getElementById("featureSelector").style.display = "block";
        document.getElementById("featureButtons").style.display = "block";
    }

    feature = element.value;
    $("#selectedFeatures").tagEditor("addTag", feature);

    if (!targetFeatureShown) {
        document.getElementById("targetFeatureDiv").style.display = "block";
    }
}

function modelSubmit() {
    var model = document.getElementById("forecasterSelection").value;

    var category = document.getElementById("indexCategory").value;
    var index = document.getElementById("indexName").value;
    var stocks = $("#selectedTickers").tagEditor("getTags")[0].tags;

    var lookBack = document.getElementById("lookBack").value;
    var forecast = document.getElementById("forecast").value;
    var features = $("#selectedFeatures").tagEditor("getTags")[0].tags;
    var targetFeature = document.getElementById("targetFeature").value;
    var modelName = document.getElementById("forecasterName").value;

    var payload = new Object();
    console.log(allForecasters[model])
    payload["model"] = model;
    payload["moduleLoc"] = allForecasters[model]["moduleLoc"]
    payload["category"] = category;
    payload["index"] = index;
    payload["tickers"] = stocks;
    payload["lookBack"] = lookBack;
    payload["forecast"] = forecast;
    payload["features"] = features;
    payload["targetFeature"] = targetFeature;
    payload["modelName"] = modelName;

    // TODO:
    /*
    1. Implement form validation
    2. Change jQuery tag editor to something better
    */

    sendPayload(payload, "/createForecaster");
    alert("Forecaster creation job has been sent, check your telegram " +
          "bot for updates. Once model creation in done, it will " +
          "automatically show up in myForecasters.")
}

function getPlotForTicker(modelLoc) {
    var ticker = document.getElementById("savedModelSelection").value;
    var plotType = document.getElementById("plotType").value;
    var numDays = document.getElementById("numDays").value;
    var payload = new Object();
    payload["modelLoc"] = modelLoc;
    payload["ticker"] = ticker;
    payload["plotType"] = plotType;
    payload["numDays"] = numDays;
    payload["type"] = modelType;
    sendPayload(payload, "/getPlot", type="POST", success=embedPlot);
}

function embedPlot(e) {
    if (e.hasOwnProperty("error")) {
        alert(e.error);
    }
    else {
        Plotly.react("plot", e.data, e.layout, {scrollZoom: true, dragmode: "pan"});
    }
}

function subscribeAgentToggle(element, agentName, modelParams) {
    console.log(modelParams);
    var subscribeButton = element;
    var payload = new Object();
    if (subscribeButton.checked == true) {
        payload["agentName"] = agentName;
        payload["savePath"] = modelParams["savePath"];
        payload["subscribe"] = 1;
    }
    else {
        payload["agentName"] = agentName;
        payload["savePath"] = modelParams["savePath"];
        payload["subscribe"] = 0;
    }
    sendPayload(payload, "/toggleAgentSubscription", type="POST")
}

function sendPayload(payload, url, type="POST", success=handleSuccess, error=handleFailure) {
    $.ajax({
        type: type,
        contentType: "application/json",
        data: JSON.stringify(payload),
        dataType: "json",
        url: url,
        success: function (e) {
            success(e);
        },
        error: function (e) {
            error(e);
        }
    });
}

function handleSuccess(e) {
    console.log(e);
}

function handleFailure(e) {
    console.log(e);
}