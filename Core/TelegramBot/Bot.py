import json
from telegram.ext import Updater, CommandHandler, MessageHandler
import warnings
from Core.TelegramBot.CustomFilters import RetrainReplyFilter
from Utils import RequestHandler as rh
from threading import Thread

with open("telegramAPIData.json") as f:
    apiData = json.load(f)


def init():
    """Method to initialize bot

    Assigns all handlers, and filters. It also loads the root information
    """
    global root, updater

    updater = Updater(apiData["apiKey"], use_context=True)

    if "rootID" not in apiData:
        message = ("Root user wasn't set, please follow"
                   " instructions in subscriptions page")
        warnings.warn(message)
        root = None
    else:
        root = apiData["rootID"]

    initCustomFilters()
    assignHandlers()


def initCustomFilters():
    """Method to initialize all custom filters
    """
    global retrainFilter

    retrainFilter = RetrainReplyFilter()


def assignHandlers():
    """Method to attach handlers to their methods
    """
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("iamroot", iAmRoot))
    dp.add_handler(MessageHandler(retrainFilter, retrainLastModel))


def start(update, context):
    """Method to handle /start command

    /start command is used to initiate a chat with the StockMate bot
    """
    chatID = update["message"]["chat"]["id"]
    if root is None:
        suffix = ("Looks like the root user isn't set. Send "
                  "/iamroot to make yourself the root")
    elif chatID == root:
        suffix = ("If you have subscribed to any trading agents, "
                  "I will send you updates on when to buy and sell."
                  "Also, I will give you updates on any train jobs"
                  "you may have set")
    elif chatID != root:
        suffix = ("Looks like you are not the root user. If you want"
                  " to become root, click on the reset root button on"
                  "your subscriptions page, and send /iamroot to me")

    message = "Hi, this is your StockMate bot. {}"
    update.message.reply_text(message.format(suffix))


def iAmRoot(update, context):
    """Method to handle /iamroot command.

    /iamroot is used to make the person texting the root user of the chatbot
    """
    chatID = update["message"]["chat"]["id"]
    if root is None:
        message = ("Alright, you have now been set as the root."
                   "You will now get updates on train jobs, "
                   "and agent subscriptions")
        saveRoot(chatID)
    elif root == chatID:
        message = ("You are already the root")
    elif root != chatID:
        message = ("Looks like you are not the root user. If you want"
                   " to become root, click on the reset root button on"
                   "your subscriptions page, and try again")
    update.message.reply_text(message)


def saveRoot(chatID):
    """Method to save the rootID locally

    Parameters
    ----------
    chatID: str
        The chatID of the user
    """
    global root
    root = chatID

    with open("telegramAPIData.json", "w+") as f:
        apiData["rootID"] = chatID
        json.dump(apiData, f)


def sendMessage(message):
    """A method to send any message to the root

    Parameters
    ----------
    message: str
        The message to be sent to the root
    """
    updater.bot.send_message(chat_id=root, text=message)


def resetRoot():
    """A method that used to reset the root for the bot

    This method will change the rootID to none. In effect, no message will
    be sent to anyone, and any commands sent to the chatbot will not be
    executed.
    """
    global root
    root = None
    sendMessage("Root has been reset. You are not the root anymore")


def retrainLastModel(update, context):
    """A method to retrain a model when the root replies with "yes"
    """
    if update["message"].text.lower() in ["yes", "y"]:
        Thread(rh.retrainForecaster()).start()
    elif update["message"].text.lower() in ["no", "n"]:
        update.message.reply_text("Okay. Saving the model now")


def startListening():
    """Method to start the bot
    """
    updater.start_polling()


if __name__ == "__main__":
    init()
    updater.start_polling()
