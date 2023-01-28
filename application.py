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
import shutil

from chatgpt_prompt import chatgpt_response
from transcribe_sd_record import transcribe_recording
from cocqui_tts import tts


from PyQt6.QtWidgets import (QMainWindow, QApplication, QLabel, 
                             QPushButton, QTextEdit)
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

        self.inputText = self.findChild(QTextEdit, 'textEditInput')

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
        """
        Method to start recording audio on button push
        """
        if self.thread.isRunning():
            self.recorder.start()
        else:
            self.thread.start()
    
    def stop(self):
        """
        Method to stop recording on button push and begins generating response
        """
        if self.thread.isRunning():
            self.recorder.stop()
        
        self.__responder_worker()
        #self.play_audiofile()
    
    def closeEvent(self, event):
        self.recorder.stop(abort=True)
        self.thread.quit()
        self.thread.wait()
    
    def __responder_worker(self):
        """
        Private helper method for the execution of the response
        """
        self.responder_thread = QThread(self)
        self.responder = Responder()
        self.responder.moveToThread(self.responder_thread)
        self.responder_thread.started.connect(self.responder.start)
        self.responder.text_response_input[str].connect(self.__responder_text_input)
        self.responder.text_response_output[str].connect(self.__responder_text_output)
        self.responder.finished.connect(self.responder_thread.quit)

        self.responder.finished.connect(self.responder.deleteLater)
        self.responder_thread.finished.connect(self.responder_thread.deleteLater)
        self.responder_thread.start()

    def __responder_text_input(self, value):
        self.inputText.append(value)

    def __responder_text_output(self, value):
        self.outputText = self.findChild(QTextEdit, 'textEditOutput')
        self.outputText.insertPlainText(value)

class Responder(QObject):
    """
    Do all the ML stuff here

    Args:
        QObject (_type_): _description_
    """
    finished = pyqtSignal()
    text_response_input = pyqtSignal(str)
    text_response_output = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.basedir = os.getcwd()

    def start(self):
        """
        Method to execute everything
        """
        self.text_to_speech()
        self.remove_temp()
        self.play_response()
        self.finished.emit()
    
    def text_to_speech(self):
        """
        Transcibe the recording, pass to chatGPT and synthesize the response
        """
        text_transcribed = transcribe_recording()
        self.text_response_input.emit(text_transcribed)
        text_response = chatgpt_response(text_transcribed)
        self.text_response_output.emit(text_response)
        print("ChatGPT responded!")
        out_path = os.path.join(self.basedir, "data", "output","temp.wav")
        tts(text_response, model_name = "en/ljspeech/vits", output_name = out_path)
        print("Response saved!")
        
    def remove_temp(self):
        """
        Remove temp files
        """
        # Remove temp files and folder after execution
        input_audio_dir = os.path.join(self.basedir,"data","input")
        vad_dir = os.path.join(input_audio_dir, "vad_chunks")
        os.remove(input_audio_dir + "/" +os.listdir(input_audio_dir)[0]) # remove files
        shutil.rmtree(vad_dir) #remove dir

    def play_response(self):
        """
        Play the synthesized and saved response
        """
        print("AUDIO BEGINNING")
        output_audio_path = os.path.join(basedir, "data", "output", "temp.wav")
        data, fs = sf.read(output_audio_path, dtype='float32')  
        sd.play(data, fs)
        sd.wait()  # Wait until file is done playing


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