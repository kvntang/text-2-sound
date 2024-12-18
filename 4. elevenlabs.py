# import requests

# # Config
# API_KEY = "sk_06a3db7d2cc74f9434402e981253414ca45e7262b10b8b8d"
# ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/standard"

# # Request Data
# text = "This is an example of ElevenLabs text-to-speech synthesis."

# headers = {
#     "xi-api-key": API_KEY,
#     "Content-Type": "application/json"
# }

# payload = {
#     "text": text,
#     "voice_settings": {
#         "stability": 0.75,
#         "similarity_boost": 0.9
#     }
# }

# # Make API Call
# response = requests.post(ELEVENLABS_URL, headers=headers, json=payload)

# # Save the Audio File
# if response.status_code == 200:
#     with open("output.wav", "wb") as f:
#         f.write(response.content)
#     print("Audio generated and saved as output.wav")
# else:
#     print(f"Failed! Status: {response.status_code}, Message: {response.text}")

from elevenlabs import ElevenLabs

client = ElevenLabs(
    api_key="sk_06a3db7d2cc74f9434402e981253414ca45e7262b10b8b8d",
)

# Generate speech
audio_generator = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    output_format="mp3_44100_128",
    text="In the heart of a misty, rain-soaked forest, where the canopy drips like a thousand tiny waterfalls, lived a clever little frog named Fynn. His emerald skin shimmered with droplets, and his golden eyes sparkled with curiosity. The forest was a place of enchantment, where the rain sang secrets to the roots and the wind whispered ancient tales through the ferns. But on this particular stormy evening, as the rain poured harder than ever before, Fynn felt a strange tug in his chest—a call from the heart of the forest that promised adventure, danger, and perhaps, a destiny far greater than any frog could ever imagine.",
    model_id="eleven_multilingual_v2",
)

# Save the audio to a file
output_file = "narration2.mp3"

with open(output_file, "wb") as file:
    for chunk in audio_generator:
        file.write(chunk)

print(f"Audio saved as {output_file}")