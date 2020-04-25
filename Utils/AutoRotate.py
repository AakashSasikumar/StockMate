import json
import requests
import os
from lxml.html import fromstring
import datetime
import random


def init(apiKey, apiUsageData="APIKeyUsageData.json", dailyLimit=500):
    global proxies, apiDataDict, apiKeys, dayLimit, apiDataDictLoc

    apiDataDictLoc = apiUsageData
    dayLimit = dailyLimit

    apiKeys = apiKey
    proxies = getProxies(len(apiKeys))

    fileName = apiUsageData.split("/")[-1]
    fileLoc = "/".join(apiUsageData.split("/")[:-1])
    if fileLoc == "":
        fileLoc = "./"
    apiDataDict = {}
    if fileName not in os.listdir(fileLoc):
        initApiDataDict()
    else:
        loadAPIDict(apiUsageData)


def initApiDataDict():
    global apiDataDict
    for apiKey in apiKeys:
        apiDataDict[apiKey] = {}
        apiDataDict[apiKey]["lastSaved"] = datetime.datetime.now()
        apiDataDict[apiKey]["timesUsed"] = 10


def loadAPIDict(apiUsageData):
    global apiDataDict
    with open(apiUsageData) as f:
        apiDataDict = json.load(f)
    for apiKey in apiKeys:
        apiDataDict[apiKey]["lastSaved"] = \
            datetime.datetime.strptime(apiDataDict[apiKey]["lastSaved"],
                                       "%Y-%m-%d %H:%M")
        if datetime.datetime.now() - apiDataDict[apiKey]["lastSaved"] \
           >= datetime.timedelta(hours=24):
            apiDataDict["timesUsed"] = 0


def getKeyAndProxy():
    global apiDataDict, proxies
    for i, apiKey in enumerate(apiKeys):
        apiData = apiDataDict[apiKey]
        if apiData["timesUsed"] < dayLimit:
            for j, proxy in enumerate(proxies):
                proxyData = proxies[proxy]
                if proxyData["timesUsed"] < dayLimit:
                    apiData["timesUsed"] += 1
                    proxyData["timesUsed"] += 1
                    apiData["lastSaved"] = datetime.datetime.now()
                    saveApiDataDict()
                    return apiKey, proxy
                if j == len(proxies):
                    # TODO: get set of new proxies
                    pass
        if i == len(apiKeys):
            message = "APIKeys have all reached day limits"
            raise Exception(message)


def saveApiDataDict():
    toSaveData = apiDataDict.copy()
    for key in toSaveData:
        data = toSaveData[key]
        data["lastSaved"] = str(datetime.datetime.now())[:-10]
    with open(apiDataDictLoc, "w+") as f:
        json.dump(apiDataDict, f)


def getProxies(num=-1):
    # TODO:
    # Add support for num > 20
    proxyDict = {}
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    for i in parser.xpath('//tbody/tr')[:num]:
        proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
        proxyDict[proxy] = {"timesUsed": 0}
    return proxyDict
