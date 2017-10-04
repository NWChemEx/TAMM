input_fname = None


def setFilename(fname):
    global input_fname
    input_fname = fname


def clearFilename():
    global input_fname
    input_fname = None


class InputError(Exception):

    def __init__(self, message, line_no=None):
        Exception.__init__(self)
        self.message = message
        self.line_no = line_no

    def __str__(self):
        if not input_fname:
            return 'error: ' + self.message
        elif not self.line_no:
            return input_fname + ': ' + self.message
        else:
            return input_fname + ':' + str(self.line_no) + ': ' + self.message


class FrontEndError(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message

    def __str__(self):
        return 'front-end error: ' + self.message
