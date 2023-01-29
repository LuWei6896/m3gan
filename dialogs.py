# This script is for dialog boxes
# dialogs dev uses basedir = os.getcwd()
# dialogs uses basedir = os.path.dirname(sys.argv[0])
from PyQt6.QtWidgets import (QDialog, QLabel, QDialogButtonBox, QTextBrowser, QPushButton)
from PyQt6 import uic
from PyQt6.QtCore import *
import os
import sys
from authoring.author import Author
from authoring.author_metadata import program_metadata
import subprocess as sp


class aboutDialog(QDialog):
    """
    Class to handle the main window for the application
    """

    def __init__(self):

        super(aboutDialog, self).__init__()  # Inherit from QDialog

        # load ui for the main window
        basedir = os.path.dirname(sys.argv[0])
        ui_dir = os.path.join(basedir, "ui", "dialogs","aboutDialog.ui")
        uic.loadUi(ui_dir, self)

        # Definitions here, widgets, variables etc.,
        self.ok_button = self.findChild(QDialogButtonBox, 'okButtonBox')
        self.ok_button.clicked.connect(self.close)

        self.metadata = self.findChild(
            QLabel, 'appMetaDataLabel')  # Print file location
        self.metadata.setText(self.__get_metadata())

        # Show the app
        self.show()

    def __get_metadata(self):
        """
        Get metadata from python module

        Returns:
            metadat_str (str): String metadata for the application
        """
        author = Author(**program_metadata)
        attrs = vars(author)
        metadata_str = "".join("%s %s\n" % item for item in attrs.items())
        return metadata_str


class helpDialog(QDialog):
    """
    Open up the README.txt or README.md as a dialog box
    """
    def __init__(self):
        super(helpDialog, self).__init__() # Inherit from QDialog
        # load ui for the main window
        basedir = os.path.dirname(sys.argv[0])
        ui_dir = os.path.join(basedir, "ui", "dialogs","helpDialog.ui")
        uic.loadUi(ui_dir, self)

        # Definitions here, widgets, variables etc.,
        self.ok_button = self.findChild(QDialogButtonBox, 'okButtonBox')
        self.ok_button.clicked.connect(self.close)

        self.readme = self.findChild(
            QTextBrowser, 'textBrowser')  # Print file location
        self.readme.setPlainText(self.__get_readme())
    
    def __get_readme(self):
        """
        Get the readme data
        """
        basedir = os.getcwd() #os.path.dirname(sys.argv[0])
        readme = os.path.join(basedir, "Help.txt")
        with open(readme, "r") as text:
            contents = text.read()
        return contents

class errorDialog(QDialog):
    """
    Error dialog that gets populated based on error that is raised and captured
    """
    def __init__(self, msg):
        """
        Start up the error dialog and pass in any catched errors as needed to present

        Args:
            msg (str): error message as str
        """
        super(errorDialog, self).__init__()
        basedir = os.path.dirname(sys.argv[0])
        ui_dir = os.path.join(basedir, "ui", "errorDialog.ui")
        uic.loadUi(ui_dir, self)
        # uic.load_ui(os.path.join(ui_dir,'errorDialog.ui'), self)

        self.ok_button = self.findChild(QDialogButtonBox, 'okButtonBox')
        self.ok_button.clicked.connect(self.close)

        self.error_message = self.findChild(QTextBrowser, 'textBrowser')
        self.error_message.setPlainText(self.__set_error_message(msg))
    
    def __set_error_message(self, msg: str):
        """
        Set error message as needed
        This method is very basic atm. Can be edited as needed to
        transform the message as required.
        Args:
            msg (str) : error message as string
        """
        return msg
