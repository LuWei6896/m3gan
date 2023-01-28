import sys
import os
from datetime import datetime
import time
import logging
import subprocess as sp
import soundfile as sf
import sounddevice as sd
import tempfile
import queue

from chatgpt_prompt import chatgpt_response
from transcribe_sd_record import transcribe_recording
from cocqui_tts import tts


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


        # Worker
        self.thread = QThread(self)
        self.recorder = Recorder()
        self.recorder.moveToThread(self.thread)
        self.thread.started.connect(self.recorder.start)
        self.recorder.finished.connect(self.thread.quit)

        self.startButton.clicked.connect(self.start)
        self.stopButton.clicked.connect(self.stop)

        self.show()

    
    def start(self):
        if self.thread.isRunning():
            self.recorder.start()
        else:
            self.thread.start()
    
    def stop(self):
        if self.thread.isRunning():
            self.recorder.stop()
        
        responder = Responder()
    
    def closeEvent(self, event):
        self.recorder.stop(abort=True)
        self.thread.quit()
        self.thread.wait()

class Responder(QObject):
    """
    Do all the ML stuff here

    Args:
        QObject (_type_): _description_
    """
    def __init__(self):
        text_transcribed = transcribe_recording()
        text_response = chatgpt_response(text_transcribed)
        print("ChatGPT responded!")
        out_path = os.path.join(os.getcwd(), "data", "output","temp.wav")
        tts(text_response, model_name = "en/ljspeech/vits", output_name = out_path)
        print("Response saved!")


class Recorder(QObject):
    """
    For Recording

    Args:
        QObject (_type_): _description_
    """
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._paused = True
        self.q = queue.Queue()


    def start(self):
        self._paused = False
        if not(self._paused):
            self.start_recording()
        else:
            print('Continue')
    
    def stop(self, *, abort = False):
        self._paused = True
        if abort:
            print('Abort')
        else:
            print('Pause')

    def callback(self, indata, frames, time ,status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def start_recording(self):
        """_summary_
        """
        self._paused = False
        if not(self._paused):
            print("Beginning")
            channels = 1
            device_info = sd.query_devices(1,'input')
            print(f"\n{device_info}")
            samplerate = int(device_info['default_samplerate'])
            print(f"\n{samplerate}")
            temp_fname = tempfile.mktemp(prefix='delme_', suffix='.wav', dir='./data/input')

            with sf.SoundFile(temp_fname, mode='x', samplerate=samplerate,
                        channels=channels) as file:
                with sd.InputStream(samplerate=samplerate, channels=channels, callback=self.callback):
                    while not(self._paused):
                        file.write(self.q.get())

        self.stop(abort=True)
        self.finished.emit()
        print('\nRecording finished: ' + repr(temp_fname))


    def stop_recording(self):
        """_summary_
        """
        self.stop_flag = 1

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