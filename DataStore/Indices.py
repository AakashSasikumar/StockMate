from selenium import webdriver
import time
import os
import json
import datetime
import sys
import subprocess


def init():
    global saveLocation
    saveLocation = "DataStore/Indices/"

    # handle system arguments
    if len(sys.argv) > 2:
        operation = sys.argv[2]
        handleOperation(operation)


def handleOperation(operation):
    if operation == "updateIndices":
        subprocess.Popen("python3 Indices.py updateIndices")


def initBrowser(baseURL="https://www1.nseindia.com/live_market/dynaContent/live_watch/live_index_watch.htm"):
    # browser = webdriver.Firefox(executable_path="DataStore/geckodriver")
    browser = webdriver.PhantomJS(executable_path="DataStore/phantomjs")
    return baseURL, browser


def updateIndices():
    categorySkipList = ["Fixed Income Indices :"]
    indexSkipList = ["NIFTY 100", "NIFTY 200",
                     "NIFTY 500", "NIFTY MIDCAP 50",
                     "NIFTY MIDCAP 100", "NIFTY SMLCAP 100",
                     "NIFTY50 TR 2X LEV", "NIFTY50 PR 2X LEV",
                     "NIFTY50 TR 1X INV", "NIFTY50 PR 1X INV",
                     "NIFTY50 DIV POINT", "INDIA VIX"]

    indices = {"type": {}}
    global browser
    baseURL, browser = initBrowser()
    browser.get(baseURL)
    rows = browser.find_elements_by_tag_name("tr")
    skipCategory = False
    for row in rows:
        # print(row.get_attribute("innerHTML"))
        if "Indices" in row.text:
            indexCategory = row.text.strip()
            if "Income" in indexCategory:
                print("asdf")
            if indexCategory in categorySkipList:
                skipCategory = True
                continue
            indices["type"][indexCategory] = {}
            skipCategory = False
        else:
            if skipCategory:
                continue
            columns = row.find_elements_by_tag_name("td")
            if len(columns) > 0:
                indexName = columns[0].text.strip()
                if indexName in indexSkipList:
                    continue
                indexLink = columns[0].find_element_by_tag_name("a").get_attribute("href").strip()
                constituents = getConstituents(indexLink, indexName)
                if not constituents:
                    continue
                else:
                    indices["type"][indexCategory][indexName] = {}
                    indices["type"][indexCategory][indexName]["constituents"] = constituents
    browser.close()
    writeIndexData(indices)


def getConstituents(link, indexName):
    _, browser = initBrowser(baseURL=link)
    browser.get(link)
    time.sleep(3)
    constituentElements = browser.find_elements_by_tag_name("tr")
    tickers = []
    for i, constituent in enumerate(constituentElements[2:]):
        consColumns = constituent.find_elements_by_tag_name("td")
        if len(consColumns) > 0:
            ticker = consColumns[0].text.strip()
            if i == 0:
                if ticker == indexName:
                    continue
                else:
                    browser.close()
                    return []
            tickerLink = consColumns[0].find_element_by_tag_name("a")
            tickerLink = tickerLink.get_attribute("href")
            tickers.append(ticker)
    browser.close()
    return tickers


def writeIndexData(indices):
    global saveLocation
    today = datetime.date.today()
    fileName = "NSE-Indices-{}.json".format(today)
    with open(saveLocation + fileName, "w+") as f:
        json.dump(indices, f)


def getDate(fileName):
    indexDate = "-".join(fileName.split("-")[2:])[:-5]
    indexDate = datetime.datetime.strptime(indexDate, "%Y-%m-%d").date()
    return indexDate


def getIndices(date=None):
    if date:
        pass
    else:
        currentDate = datetime.date.today()
        indexData = sorted(os.listdir(saveLocation),
                           key=lambda x: getDate(x))[-1]
        indexDate = getDate(indexData)
        daysDelta = currentDate - indexDate
        if daysDelta.days > 30:
            # figure out how to make this properly parallel
            handleOperation("updateIndices")
        return json.load(open(saveLocation + indexData))


if __name__ == "__main__":
    init()
    # updateIndices()
    print(getIndices())
