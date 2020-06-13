from selenium import webdriver
import time
import os
import json
import datetime


class NSEIndices():
    """A class with functionality to scrape and save data of all NSE indices

    Attributes
    ----------
    saveLocation: str
        The location at which the Index data is to be saved
    projectRoot: str
        The root folder of the project. This is used to make sure that data
        is saved in the proper location no matter where this module is called
        from within the project
    saveLocation: str
        The location at which the data is to be stored
    """
    def __init__(self, saveLocation="DataStore/Indices/NSE", updateEvery=180):
        self.updateEvery = updateEvery
        self.projectRoot = self.getProjectRoot()
        self.saveLocation = self.projectRoot + "DataStore/Indices/"
        self.initDirectories()

    def getProjectRoot(self):
        """Returns the root directory of the folder

        This function is used to make sure that this module works even
        if used by other modules in other directories.

        Returns
        -------
        currentPath:
            The root path of the project
        """
        currentPath = os.getcwd()
        while(True):
            if "DataStore" in os.listdir(currentPath):
                break
            currentPath = "/".join(currentPath.split("/")[:-1])
        return currentPath + "/"

    def initDirectories(self):
        """Create directory structure if not made already
        """

        lsDataStore = os.listdir(self.projectRoot + "DataStore/")
        if "Indices" not in lsDataStore:
            os.mkdir(self.projectRoot + "DataStore/Indices")
            os.mkdir(self.projectRoot + "DataStore/Indices/NSE")

    def initBrowser(self, baseURL="https://www1.nseindia.com/live_market/dynaContent/live_watch/live_index_watch.htm",
                    browserType="phantom"):
        """Returns the selenium browser and the base URL

        Returns
        -------
        baseURL: str
            The base URL from which the scraping starts
        browser: selenium.webdriver.<browserType>
        """
        if browserType == "phantom":
            browser = webdriver.PhantomJS(executable_path=self.projectRoot+"DataStore/phantomjs")
        elif browserType == "firefox":
            browser = webdriver.Firefox(executable_path=self.projectRoot+"DataStore/geckodriver")

        return baseURL, browser

    def updateIndices(self):
        """Scrapes all the NSE Indices and saves their data
        """
        categorySkipList = ["Fixed Income Indices"]
        indexSkipList = ["NIFTY 100", "NIFTY 200",
                         "NIFTY 500", "NIFTY MIDCAP 50",
                         "NIFTY MIDCAP 100", "NIFTY SMLCAP 100",
                         "NIFTY50 TR 2X LEV", "NIFTY50 PR 2X LEV",
                         "NIFTY50 TR 1X INV", "NIFTY50 PR 1X INV",
                         "NIFTY50 DIV POINT", "INDIA VIX"]

        indices = {"type": {}}
        baseURL, browser = self.initBrowser()
        browser.get(baseURL)
        rows = browser.find_elements_by_tag_name("tr")
        skipCategory = False
        for row in rows:
            if "Indices" in row.text:
                indexCategory = row.text.strip()[:-2]
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
                    indexLink = columns[0].find_element_by_tag_name(
                        "a").get_attribute("href").strip()
                    constituents = self.scrapeConstituents(indexLink,
                                                           indexName)
                    if not constituents:
                        continue
                    else:
                        indices["type"][indexCategory][indexName] = {}
                        indices["type"][indexCategory][indexName]["constituents"] = constituents
        browser.close()
        self.writeIndexData(indices)

    def scrapeConstituents(self, link, indexName):
        """Scrapes another page to get the list of constituents

        The list of all Indices is available in the baseURL,
        to get the constituents of each index, another page has to be
        scraped

        Parameters
        ----------
        link: str
            url of page to be scraped
        indexName: str
            name of the index for which we are scraping the constituents

        Returns
        -------
        tickers: list(str)
            A list of all the tickers under the index
        """
        _, browser = self.initBrowser(baseURL=link)
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

    def writeIndexData(self, indices):
        """Writes the scraped data for all the indices

        The file name convention is as follows,
        NSE-Indices-yyyy-mm-dd.json where the date is the the date of writing
        the data.

        Parameters
        ----------
        indices: dict
            A dictionart of all the indices and their constituents
        """
        today = datetime.date.today()
        fileName = "NSE-Indices-{}.json".format(today)
        with open(self.saveLocation + fileName, "w+") as f:
            json.dump(indices, f)

    def getDate(self, fileName):
        """Returns the date on which the Indices were saved.

        This method is used to check how old the data is, and an
        appropriate action is taken.

        Parameters
        ----------
        fileName: str
            The name of the index data file

        Returns
        -------
        indexData: datetime.datetime
            The date on which the data was saved
        """
        indexDate = "-".join(fileName.split("-")[2:])[:-5]
        indexDate = datetime.datetime.strptime(indexDate, "%Y-%m-%d").date()
        return indexDate

    def getIndices(self):
        """Returns the latest NSE Indices data

        Returns:
        indexData: dict
            The dict containing data of all NSE indices
        """

        currentDate = datetime.date.today()
        indexData = sorted(os.listdir(self.saveLocation),
                           key=lambda x: self.getDate(x))[-1]
        indexDate = self.getDate(indexData)
        daysDelta = currentDate - indexDate
        if daysDelta.days > self.updateEvery:
            # TODO:
            # Implement a background task to update the indices data
            pass
        return json.load(open(self.saveLocation + indexData))
