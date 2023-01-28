Note: This is a work in progress (WIP)
Think Siri but with your choice of voice!
This application will take your prompts and repond verbally through a variety of sound options

High Level Process Flow:

1. Record speech from microphone on button push to start recording
2. Save recorded speech into a temp to stop recording
   \*Steps 3 - 6 begins upon pressing the "Stop" button
3. Transcribe saved speech in temp
4. Pass transcribed speech to ChatGPT
5. Store ChatGPT response
6. Pass response to Cocqui-TTS for speech synthesis (Preferred would be Microsoft VALL-E if available)
7. Read saved TTS wav file and playback as audio
8. Perform temporary file cleanup

Background:
I watched the movie M3gan (hence the repo name) which prompted me to build this by chaining different models to reach an outcome of AI generated response to human input via speech.

My aim for this is to have something like a conversational AI with ChatGPT as the responder.

Requirements:

- Models:
  - OpenAI Whisper medium.en.pt for transcription
  - OpenAI member with key generated to access chatGPT via API call for response
  - en/ljspeech/vits model from cocqui-tts (available on huggingface) for speech to text
