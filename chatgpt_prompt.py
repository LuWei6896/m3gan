import openai
from secrets.credentials import oai_credentials

def chatgpt_response(transcribed_text):
    openai.api_key = oai_credentials
    transcribed_text = transcribed_text

    model_engine = "text-davinci-002"
    prompt = (f"transcriber: {transcribed_text}")

    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    return response