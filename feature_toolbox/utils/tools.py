from os.path import isdir

class database_loader(object):
    """ Class for handling the database path. Can have some real functionality later. Now its just a wrapper for the filepath"""
    def __init__(self, path):
        self.filename=path
