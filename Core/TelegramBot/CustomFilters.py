from telegram.ext import BaseFilter


class retrainReplyFilter(BaseFilter):
    def __init__(self):
        self.toggleRetrain = False
        self.options = ["yes", "no", "y", "n"]

    def filter(self, message):
        print(self.toggleRetrain)
        if message.text.lower() in self.options:
            if self.toggleRetrain:
                return True
        return False
