from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import os
import pdb

# Running a multi-speaker and multi-lingual model
#model_name = TTS.list_models()[0]

# Listening to https://www.youtube.com/watch?v=HojuVmW5LUI&ab_channel=Thorsten-Voice
# I have chosen ljspeech/vits given its sound and high RTF score
# tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

def tts(text: str, model_name: str, output_name: str,speaker_idx: str=None):
    """_summary_

    Args:
        text (str): _description_
        model_name (str): _description_
        output_name (str): _description_
        speaker_idx (str, optional): _description_. Defaults to None.

    Raises:
        NameError: _description_
    
    Reference:
        https://huggingface.co/spaces/coqui/CoquiTTS/blob/main/app.py
    """
    manager = ModelManager()
    # download model
    model_path, config_path, model_item = manager.download_model(f"tts_models/{model_name}")
    vocoder_name:str = model_item["default_vocoder"]
    # download vocoder
    vocoder_path = None
    vocoder_config_path = None
    if vocoder_name is not None:
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
    # init synthesizer
    synthesizer = Synthesizer(
        model_path, config_path, None, None, vocoder_path, vocoder_config_path,
    )
    # synthesize
    if synthesizer is None:
        raise NameError("model not found")
    wavs = synthesizer.tts(text, speaker_idx)

    # return output
    print(output_name)
    synthesizer.save_wav(wavs, output_name)

if __name__ == "__main__":
    text = "This sentence has been generated by a speech synthesis system."
    model_name = "en/ljspeech/vits"
    basedir = os.getcwd()
    AUDIODIR = os.path.join(basedir, "data")
    output_name = "test"
    output_format = ".wav"
    output_path = os.path.join(basedir, "output",output_name + output_format)
    tts(text, model_name)