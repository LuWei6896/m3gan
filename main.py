import sys
import os
from datetime import datetime
import time
import logging
import subprocess as sp
import soundfile as sf

from PyQt6.QtWidgets import (QMainWindow, QApplication, QLabel,
                             QPushButton)
from PyQt6.QtGui import QAction
from PyQt6 import uic
from PyQt6.QtCore import *

def except_hook(cls, exception, traceback):
    """
    Capture exceptions and prevent GUI from auto closing upong errors

    Args:
        exception (_type_): _description_
        traceback (_type_): _description_
    """
    sys.__excepthook__(cls, exception, traceback)


class mainUI(QMainWindow):
    """
    Class to handle the main window for the application
    """

    def __init__(self):

        super(mainUI, self).__init__()  # Inherit from QMainWindow

        # load ui for the main window
        basedir = os.getcwd() # set to os.getcwd() if running not in pyinstaller mode
        # ui_dir = os.path.join(basedir, 'ui', 'mainWindow.ui')
        ui_dir = os.path.join(basedir, 'ui', 'mainWindow.ui')
        
        uic.loadUi(ui_dir, self)

        self.startButton = self.findChild(QPushButton, 'startButton')
        self.stopButton = self.findChild(QPushButton, 'stopButton')

if __name__ == '__main__':
    sys.excepthook = except_hook
    app = QApplication(sys.argv)  # Need this to start up the app

    # Add styles
    basedir = os.getcwd() # os.getcwd()
    # style_dir = os.path.join(basedir, 'styles', 'style.qss')
    style_dir = os.path.join(basedir, 'styles', 'styles.qss') # dev

    with open(style_dir, 'r') as f:
        style = f.read()
    app.setStyleSheet(style)
    mainWindow = mainUI()
    app.exec()  # Execute the app