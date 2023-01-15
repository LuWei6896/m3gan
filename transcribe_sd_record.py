from whisper_main import Transcriber
import os

basedir = os.getcwd()
VAD_MODEL_DIR = os.path.join(basedir, "models", "silero-vad")
AUDIO_DIR = os.path.join(basedir, "data", "input")
ffmpeg_path = os.path.join(basedir, "ffmpeg", "ffmpeg.exe")
full_audio_path = os.path.join(AUDIO_DIR, "test_1.wav")
transcriber = Transcriber(*[VAD_MODEL_DIR, full_audio_path, AUDIO_DIR]) # Store the object in mem
transcriber._run(ffmpeg_path, basedir, output_checker=0) 
text_result_list = transcriber.getResult(output_checker = 0)
transcribed_text = " ".join(text_result_list)
