from whisper_module import whisper
import csv
import os
import sys
import pdb
from datetime import datetime
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pathlib
import torch
from tqdm import tqdm
import ffmpeg
import datetime


# Run the english only model as they do better for eng only apps

# file_path = "./data/input/delta_orig.wav"


def whisper_transcribe(file_path: str, model_name: str, ffmpeg_path:str, basedir: str):
    """Main function to transcribe audio using chosen model and file

    Args:
        audio_file (str): _description_
        model_name (str): _description_
        ffmpeg_path (str): _description_
        basedir (str): _description_

    Returns:
        result (dict): Dictionary of transcription which can be accessed as needed
    """
    model_name = model_name
    print(f"File path: {file_path}\n")
    print(f"Loading model: {model_name}...\n")
    model = whisper.load_model_local(f"{model_name}.en", basedir ,in_memory=True)

    print(f"Model loaded! Beginning transcription...\n")
    result = model.transcribe(file_path, ffmpeg_path)
    print(f"Transcription complete!\n")
    return result

# Redo transcribe to csv to feed in result dict
# Then we search and check for 'segments' within, and create the list here


def transcribe_to_csv(whisper_results: dict, output_name: str):
    """_summary_
    Assuming the list contains dictionaries all with the same keys
    Then write to csv with keys as a header
    Args:
        whisper_results (dict): _description_
        output_name (str): audio_file name for automatic file name creation.
                            Should not be manual.
    """
    segments = whisper_results['segments']
    if (len(segments) == None) or (len(segments) == 0):
        print("Empty list. Please check transcription worked")
        sys.exit()
    if ".wav" in output_name:
        output_name = output_name[:-4]
        # e.g. output_name == 'result"
    myFile = open(output_name, 'w', newline='')
    writer = csv.writer(myFile)
    writer.writerow(segments[0].keys())  # e.g. segments
    for dictionary in segments:  # e.g. segment in segments
        writer.writerow(dictionary.values())
    myFile.close()
    print("Transcription successfully written to file!")


def transcribe_to_text(whisper_results: dict, output_name: str):
    """
    Transcribe to text

    Args:
        whisper_results (dict): Transcriped Whisper object
        output_name (str): _description_
    """
    transcribed_text = whisper_results['text']
    if (len(transcribed_text)) == None or (len(transcribed_text)) == 0:
        print("Empty string. Check that transcription has output")
    if ".wav" in output_name:
        output_name = output_name[:-4]
    with open(output_name, "w") as text_file:
        text_file.write(transcribed_text)

def transcribe_to_file(whisper_results, output_name):
    """
    Transcribe results to txt and csv

    Args:
        whisper_results (_type_): _description_
        output_name (_type_): _description_
    """
    transcribe_to_text(whisper_results['text'], output_name)
    transcribe_to_csv(whisper_results['segments'], output_name)


def run_whisper(file_path, ffmpeg_path, basedir):
    """
    One liner function to run whisper execution
    Can be executed standalone
    Designed to be called within the app.py from a gui

    Args:
        filepath (str): Filepath to wav
        ffmpeg_path (str): Filepath to ffmpeg exe
        basedir (str): File path to model
    """
    model_name = "medium"#"tiny" # "medium"
    #print(f"Execution started: {datetime.now()}")
    # Transcribe audio
    result = whisper_transcribe(file_path=file_path, model_name=model_name, ffmpeg_path=ffmpeg_path, basedir=basedir)
    return result


class Transcriber(object):
    """
    This class implements a new VAD ontop of Whisper Transcriber
    """
    def __init__(self, *args, **kwargs):
        self.VAD_THRESHOLD = 0.4 # Confidence threshold for VAD speech vs non-speech
        self.VAD_SR = 16000 # Sample rate to resample to
        self.head = 3200 # 0.2s head for padding chunks
        self.tail = 20800 # 1.32 tail for padding chunks
        self.chunk_threshold = 3.0 

        # Pass in directories by unpacking a list
        # List order should be in row order below
        self.SILERO_DIR = args[0] # This should be passed in from os.path.dirname(sys.argv[0]) or os.getcwd() if dev
        self.full_audio_path = args[1] # This will get passed in from a button or another class attribute
        self.AUDIO_DIR = args[2]

    def _vad_dir_creator(self):
        """
        Create directory for VAD chunks if not existing already
        """
        AUDIO_DIR = self.AUDIO_DIR
        if not os.path.exists(os.path.join(AUDIO_DIR,"vad_chunks")):
            print("Creating vad chunks directory...")
            os.mkdir(os.path.join(AUDIO_DIR,"vad_chunks"))
        
        # Store directory as we need somewhere to look for later on
        self.VAD_DIR = os.path.join(AUDIO_DIR, "vad_chunks")
        print("Directory created!")
        
    def _create_temp_audio(self):
        """
        Create a copy of the input audio from full_audio_path and write to a temp directory VAD_DIR
        """
        full_audio_path = self.full_audio_path
        VAD_DIR = self.VAD_DIR
        
        # Hardcode values to feed into ffmpeg
        # This is just a temporary copy which requires specific formatted values to ensure
        # that we get the appropriate input to the system
        ffmpeg.input(full_audio_path).output(
            VAD_DIR+"/vad_temp.wav",
            ar="16000",
            ac="1",
            acodec="pcm_s16le",
            map_metadata="-1",
            fflags="+bitexact",
        ).overwrite_output().run(quiet=True)
        print(os.path.exists(os.path.join(VAD_DIR,"vad_temp.wav")))
    
    def _load_vad(self):
        """
        Load the VAD model from local
        Store utils from Silero as methods within the transcriber class
        """
        vad_model, utils = torch.hub._load_local(
            hubconf_dir=self.SILERO_DIR, model="silero_vad_local", onnx=False
        )
        if (vad_model is not None) and (utils is not None):
            print("Model and utils loaded!")

        # Store the functions within the transcriber object for easy access
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = utils

        self.vad_model = vad_model 
    
    def _read_audio(self):
        """
        Read audio into an attribute to use
        The path here will be hardcoded as everything should be self contained
        The self-containment is from the directory and file copying from
        _vad_dir_creator() and _create_temp_audio() which if properly run
        will result in absolute file locations    
        """
        self.wav = self.read_audio("data/input/vad_chunks/vad_temp.wav", sampling_rate=self.VAD_SR)
        print("Audio loaded! at %s" % ("data/input/vad_chunks/vad_temp.wav"))
    
    def _process_timestamps(self):
        """
        Add padding, remove small gaps and overlaps from processed audio
        Result timestamps are in samples (not seconds) 
        They represent chunks of audio
        """
        t = self.get_speech_timestamps(self.wav, self.vad_model, sampling_rate=self.VAD_SR, threshold=self.VAD_THRESHOLD)    
        # Add a bit of padding, and remove small gaps
        for i in range(len(t)):
            t[i]["start"] = max(0, t[i]["start"] - self.head)  # 0.2s head -> self.head=3200
            t[i]["end"] = min(self.wav.shape[0] - 16, t[i]["end"] + self.tail)  # 1.3s tail -> self.tail = 20800
            if i > 0 and t[i]["start"] < t[i - 1]["end"]:
                t[i]["start"] = t[i - 1]["end"]  # Remove overlap
        self.timestamps = t # Store timestamps to edit
        
        print("Timestamps processed! No. of audio chunks are: {}".format(len(self.timestamps)))
        
    def _split_audio(self): 
        """
        If breaks are longer than chunk_threshold seconds, 
            split into a new audio file
        This'll effectively turn long transcriptions into many shorter ones
        
        """
        # Store the chunked audio into a matrix
        # Each row is each chunk
        u = [[]]
        for i in range(len(self.timestamps)):
            if i > 0 and self.timestamps[i]["start"] > self.timestamps[i - 1]["end"] + (self.chunk_threshold * self.VAD_SR):
                u.append([])
            u[-1].append(self.timestamps[i])
        self.chunked_audio = u # Store the matrix of chunked audio
        
        print("No. of chunked audio based on threshold: {}".format(len(self.chunked_audio)))
    
    def _merge_chunks(self):
        """
        Merge chunks and remove temp copy of original audio
        Save chunks to a directory

        """
        for i in range(len(self.chunked_audio)):
            self.save_audio(
                "data/input/vad_chunks/" + str(i) + ".wav", # Fix the hardcoded locations for path
                self.collect_chunks(self.chunked_audio[i], self.wav),
                sampling_rate=self.VAD_SR,
            )
        if len(os.listdir("data/input/vad_chunks")) != 0:
            print("Audio chunks saved!")
        os.remove("data/input/vad_chunks/vad_temp.wav") # Fix hardcoded paths
    
    def _convert_timestamps_seconds(self):
        """
        Convert timestamps into seconds format
        """
        # Go through each individual chunked audio within the matrix
        # Identify chunks and offsets for the audio through math and using 16000 SR
        # keys: start, end, chunk_start and chunk_end should be present if previous executions worked
        for i in range(len(self.chunked_audio)):
            time = 0.0
            offset = 0.0
            for j in range(len(self.chunked_audio[i])):
                self.chunked_audio[i][j]["start"] /= self.VAD_SR
                self.chunked_audio[i][j]["end"] /= self.VAD_SR
                self.chunked_audio[i][j]["chunk_start"] = time
                time += self.chunked_audio[i][j]["end"] - self.chunked_audio[i][j]["start"]
                self.chunked_audio[i][j]["chunk_end"] = time
                if j == 0:
                    offset += self.chunked_audio[i][j]["start"]
                else:
                    offset += self.chunked_audio[i][j]["start"] - self.chunked_audio[i][j - 1]["end"]
                self.chunked_audio[i][j]["offset"] = offset
        print("Timestamps converted!")
    
    def _whisper_on_chunks(self, ffmpeg_path:str, basedir: str, model_name:str = 'medium'):
        """
        Transcribe using whisper on the pre-processed chunked audio
        Whisper using the medium model

        Args:
            ffmpeg_path (str): path to ffmpeg executable e.g., os.path.join(basedir,"ffmpeg","ffmpeg.exe")
            basedir (str): Path where executable should be located e.g., os.getcwd() 
            model_name (str): Default "medium" for Whisper model size
        """
        # Load the whisper model
        model = whisper.load_model_local(f"{model_name}.en", basedir ,in_memory=True)
        task = 'transcribe'
        language = 'english'
        initial_prompt = ''

        # Transcribe each chunk using Whisper
        for i in tqdm(range(len(self.chunked_audio))):
            result = model.transcribe(
                os.path.join(self.VAD_DIR,str(i) + ".wav" ), ffmpeg_path=ffmpeg_path, 
                task=task, language=language, initial_prompt=initial_prompt
            )
            # Break if result doesn't end with severe hallucinations
            if len(result["segments"]) == 0:
                break
            elif result["segments"][-1]["end"] < self.chunked_audio[i][-1]["chunk_end"] + 10.0:
                break
        self.result = result
        if len(result) > 0:
            print("Audio successfully transcribed!")
    
    def _run(self, ffmpeg_path:str, basedir: str):
        """
        This method executes all the required steps required for Whisper

        Args:
            ffmpeg_path (str): path to ffmpeg.exe 
            basedir (str): directory to search for everyting
        """
        self._vad_dir_creator()
        self._create_temp_audio()
        self._load_vad()
        self._read_audio()
        self._process_timestamps()
        self._split_audio()
        self._merge_chunks()
        self._convert_timestamps_seconds()
        self._whisper_on_chunks(ffmpeg_path = ffmpeg_path, basedir = basedir)
            
    def whisper_csv(self, ffmpeg_path:str, basedir: str):
        """
        Seperate whisper results into csv only
        """
        self._run(ffmpeg_path = ffmpeg_path, basedir = basedir) # Execute everything needed
        segment_info = []
        result = self.result
        
        for i in tqdm(range(len(self.chunked_audio))):
            for r in result["segments"]:
                # Skip audio timestamped after the chunk has ended
                if r["start"] > self.chunked_audio[i][-1]["chunk_end"]:
                    continue
                # Skip if log prob is low or no speech prob is high
                if r["avg_logprob"] < -1.0 or r["no_speech_prob"] > 0.7: # Hardcoded thresholds
                    continue
                # Strip white spaces
                r['text'] = r['text'].strip() 
                segment_info.append(r)
                
        return segment_info
    
    def whisper_text(self, ffmpeg_path: str, basedir: str):
        """
        Separate the whisper results into text for output
        """
        self._run(ffmpeg_path = ffmpeg_path, basedir = basedir)
        text_info = []
        result = self.result

        # Post process the segments and text based using the offsets and 
        # identified chunk start and end to ensure we can remove overlaps
        for i in tqdm(range(len(self.chunked_audio))):
            for r in result["segments"]:
                # Skip audio timestamped after the chunk has ended
                if r["start"] > self.chunked_audio[i][-1]["chunk_end"]:
                    continue
                # Skip if log prob is low or no speech prob is high
                if r["avg_logprob"] < -1.0 or r["no_speech_prob"] > 0.7: # Hardcoded thresholds
                    continue                
                text_info.append(r["text"].strip())
        return text_info


# To test things
# if __name__ == '__main__':
#     basedir = os.getcwd()
#     ffmpeg_path = os.path.join(basedir,"ffmpeg","ffmpeg.exe")
#     AUDIO_DIR = os.path.join(basedir, "data", "input")
#     full_audio_path = os.path.join(basedir, "data","input","delta_orig.wav")
#     SILERO_DIR = os.path.join(basedir, "models", "silero-vad")
#     transcriber = Transcriber(*[SILERO_DIR, full_audio_path, AUDIO_DIR])

#     import pdb
#     import json
#     results = run_whisper(file_path = "data/input/delta_orig.wav", ffmpeg_path = ffmpeg_path, basedir = basedir)
#     # with open("results.json", 'w') as fp:
#     #     json.dump(results['segments'], fp)
    
#     segments_info = transcriber.whisper_csv(ffmpeg_path = ffmpeg_path, basedir = basedir)
#     # with open("segments_info.json", "w") as fp:
#     #     json.dump(segments_info, fp)
#     # text_info = transcriber.whisper_text(ffmpeg_path = ffmpeg_path, basedir = basedir)
#     #pdb.set_trace()