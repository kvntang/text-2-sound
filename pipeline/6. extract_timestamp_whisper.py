import whisper

# Load the model
model = whisper.load_model("base")

# Transcribe audio file with word-level timestamps
result = model.transcribe("narration2.mp3", word_timestamps=True)

# Words to Timestamp
target_words = ["rain", "forest", "adventure"]

# Initialize dictionary for timestamps
timestamps = {word: [] for word in target_words}

# Collect timestamps for each target word
for segment in result["segments"]:
    for word in segment["words"]:
        # Clean the word: lowercase, strip punctuation and spaces
        cleaned_word = word["word"].lower().strip(",.?! ").strip()
        
        # # Debug: print each word being checked
        # print(f"Checking word: '{cleaned_word}'")
        
        if cleaned_word in target_words:
            start_time = float(word['start'])
            end_time = float(word['end'])
            timestamps[cleaned_word].append((start_time, end_time))

# Print the dictionary
print("\nTimestamps for Target Words:")
# print(timestamps)
for key, value in timestamps.items():
    print(f"{key}: {value}")
