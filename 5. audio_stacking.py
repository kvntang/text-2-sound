from pydub import AudioSegment
from pydub.playback import play


# Load audio components
narration = AudioSegment.from_file("narration.mp3")
sound_effect1 = AudioSegment.from_file("audio_files/whistle.m4a")
sound_effect2 = AudioSegment.from_file("audio_files/meow.m4a")

# Overlay sound effects at specific times
final_audio = narration.overlay(sound_effect1, position=500)  # Add effect1 at 3s
final_audio = final_audio.overlay(sound_effect2, position=2000)  # Add effect2 at 8s

# Export final audio
final_audio.export("final_output.mp3", format="mp3")
play(final_audio)
