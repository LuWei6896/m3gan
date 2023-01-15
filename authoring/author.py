class Author(object):
    """
    Class to provide program metadata

    Args:
        object (dict): dictionary containing program metada
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)