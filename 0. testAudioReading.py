import librosa

# Load the audio file
audio, sr = librosa.load("audio2.m4a", sr=16000, mono=True)

# Calculate audio length in seconds
audio_length_seconds = len(audio) / sr

# Print the results
print(f"Loaded audio with sample rate: {sr}, length: {len(audio)} samples")
print(f"Audio length: {audio_length_seconds:.2f} seconds")
