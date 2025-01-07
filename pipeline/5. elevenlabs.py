from elevenlabs import ElevenLabs
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(
    api_key=API_KEY,
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