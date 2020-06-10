import json
from telegram.ext import Updater, CommandHandler
import warnings


with open("telegramAPIData.json") as f:
    apiData = json.load(f)

RESET_ROOT = False


def init():
    global root, updater

    updater = Updater(apiData["apiKey"], use_context=True)

    if "rootID" not in apiData:
        message = ("Root user wasn't set, please follow"
                   " instructions in subscriptions page")
        warnings.warn(message)
        root = None
    else:
        root = apiData["rootID"]

    assignHandlers()


def assignHandlers():
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("iamroot", iAmRoot))


def start(update, context):
    chatID = update["message"]["chat"]["id"]
    if root is None:
        suffix = ("Looks like the root user isn't set. Send "
                  "/iamroot to make yourself the root.")
    elif chatID == root:
        suffix = ("If you have subscribed to any trading agents, "
                  "I will send you updates on when to buy and sell."
                  "Also, I will give you updates on any train jobs"
                  "you may have set.")
    elif chatID != root:
        suffix = ("Looks like you are not the root user. If you want"
                  " to become root, click on the reset root button on"
                  "your subscriptions page, and send /iamroot to me.")

    message = "Hi, this is your StockMate bot. {}"
    update.message.reply_text(message.format(suffix))


def iAmRoot(update, context):
    chatID = update["message"]["chat"]["id"]
    if root is None:
        message = ("Alright, you have now been set as the root."
                   "You will now get updates on train jobs, "
                   "and agent subscriptions.")
        saveRoot(chatID)
    elif root == chatID:
        message = ("You are already the root.")
    elif root != chatID:
        message = ("Looks like you are not the root user. If you want"
                   " to become root, click on the reset root button on"
                   "your subscriptions page, and try again.")
    update.message.reply_text(message)


def saveRoot(chatID):
    global root
    root = chatID

    with open("telegramAPIData.json", "w+") as f:
        apiData["rootID"] = chatID
        json.dump(apiData, f)


def sendMessage(message):
    updater.bot.send_message(chat_id=root, text=message)


def resetRoot():
    global root
    root = None
    sendMessage("Root has been reset. You are not the root anymore.")


def startListening():
    updater.start_polling()


if __name__ == "__main__":
    init()
    updater.start_polling()
    # updater.idle()
