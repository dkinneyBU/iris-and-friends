import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

for idx, voice in enumerate(voices):
    print(f"{idx}: {voice.name} ({voice.id})")
