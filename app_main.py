# Testing on pre-saved recordings first

from chatgpt_prompt import chatgpt_response
from transcribe_sd_record import transcribe_recording
from cocqui_tts import tts

def main(out_path: str):
    """_summary_

    Args:
        out_path (str): _description_
    """
    text_transcribed = transcribe_recording()
    text_response = chatgpt_response(text_transcribed)
    print("ChatGPT responded!")
    tts(text_response, model_name = "en/ljspeech/vits", output_name =out_path)
    print("Response saved!")

if __name__ == "__main__":
    out_path = "data/output/temp.wav"
    main(out_path)