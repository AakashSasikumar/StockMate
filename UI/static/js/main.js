function submitAPIKey() {
    //TODO:
    // Finish implementing this..
    apiKey = document.getElementById("apiKey").value;
    console.log(apiKey)
    console.log("To be implemented...")
}

function forecasterSelected() {
    showForecasterDescription();
    showForecasterParams();
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

function showForecasterParams() {
    var selectedModel = document.getElementById("forecasterSelection").value;
    var modelParams = allForecasters[selectedModel]["params"];
    for (var i = 0; i < allForecasterParams.length; i++) {
        paramDivID = allForecasterParams[i]+"ForecasterParamDiv";
        document.getElementById(paramDivID).style.display = "none";
        if (modelParams.includes(allForecasterParams[i])) {
            document.getElementById(paramDivID).style.display = "block";
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

function featureSelected(element) {
    if (!featureSelectorMade) {
        initFeatureSelector()
        featureSelectorMade = true;
    }

    feature = element.value;
    $("#selectedFeatures").tagEditor("addTag", feature);
}