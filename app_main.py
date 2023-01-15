# Testing on pre-saved recordings first

from chatgpt_prompt import chatgpt_response
from transcribe_sd_record import transcribe_recording

def main():
    text_transcribed = transcribe_recording()
    text_response = chatgpt_response(text_transcribed)
    print(text_response)

if __name__ == "__main__":
    main()