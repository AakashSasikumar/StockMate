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
