Note: This is a work in progress (WIP)
Think Siri but with your choice of voice!
This application will take your prompts and repond verbally through a variety of sound options

High Level Process Flow:

1. Record speech from microphone
2. Save recorded speech into a temp
3. Transcribe saved speech in temp
4. Pass transcribed speech to ChatGPT
5. Store ChatGPT response
6. Pass response to Cocqui-TTS for speech synthesis (Preferred would be Microsoft VALL-E if available for SOTA)

Background:
I watched the movie M3gan (hence the repo name) which prompted me to build this by chaining different models to reach an outcome.

My aim for this is to have something like a conversational AI with ChatGPT as the responder.
