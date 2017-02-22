class BackEndError(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return 'back-end error: ' + self.message
