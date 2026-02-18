from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # dummy
)

resp = client.chat.completions.create(
    model="mistral",
    messages=[
        {"role": "system", "content": "Reply with exactly: PONG"},
        {"role": "user", "content": "Now."},
    ],
    temperature=0,
    max_tokens=3,
)

print(resp.choices[0].message.content)
