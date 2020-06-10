import json
import os
import Core.Bot.TelegramBot as tbot


def saveTelegramAPIKey(apiKey):
    saveDict = {"apiKey": apiKey}
    with open("telegramAPIData.json", "w+") as f:
        json.dump(saveDict, f)


def resetTelegramRoot():
    if "telegramAPIData.json" not in os.listdir():
        return

    with open("telegramAPIData.json") as f:
        apiData = json.load(f)

    if "rootID" not in apiData.keys():
        return

    # only saves the apiKey, and effectively removing rootID
    saveTelegramAPIKey(apiData["apiKey"])
    tbot.restRoot()
