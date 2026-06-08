import os
from openai import OpenAI

key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=key)  # uses default https://api.openai.com/v1

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "ping"}],
)
print(resp.choices[0].message.content)
