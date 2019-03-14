from logging import FileHandler, Formatter


class DebugFileHandler(FileHandler):
    """This class handles the debugging of iceberg simulations by writing variables to file.
    """
    def __init__(self, filename='debug.log', mode='w', encoding=None, delay=False):
        FileHandler.__init__(self, filename, mode, encoding, delay)
        self.formatter = Formatter('%(message)s')
        self.setFormatter(self.formatter)
