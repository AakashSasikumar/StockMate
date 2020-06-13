from telegram.ext import BaseFilter


class RetrainReplyFilter(BaseFilter):
    """Class to filter out yes/no messages for when retrain is triggered

    Attributes
    ----------
    toggleTrain: bool
        A variable indicating if retrain state is toggled
            - Retrain is done only if the chatbot is in the retrain state
            - This boolean is set by UIRequestHandler.py
    options: list
        A list of all the options the user has to reply with
    """
    def __init__(self):
        self.toggleRetrain = False
        self.options = ["yes", "no", "y", "n"]

    def filter(self, message):
        if message.text.lower() in self.options:
            if self.toggleRetrain:
                return True
        return False
