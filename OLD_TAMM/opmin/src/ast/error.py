class ASTError(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return 'abstract syntax tree error: ' + self.message
