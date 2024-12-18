import torch
from transformers import ClapProcessor, ClapModel
import faiss
from pydub import AudioSegment
from pydub.playback import play
import os

# Disable Tokenizer Parallelism at the Start
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Retrieval: Find similar audio from FAISS
def search_audio_by_text(query_text, k=1):
    # Load the FAISS index and metadata
    index = faiss.read_index("audio_index.faiss")
    audio_metadata = torch.load("audio_metadata.pt", weights_only=True)

    # Load CLAP model and processor
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")

    # Convert text query into an embedding
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)

    with torch.no_grad():
        query_embedding = model.get_text_features(**text_inputs).numpy()

    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search the FAISS index for matching audio
    distances, indices = index.search(query_embedding, k)

    # Display results
    print(f"\nSearch Results for: '{query_text}'")
    top_file_paths = []
    for i, idx in enumerate(indices[0]):
        file_path = audio_metadata[idx]
        similarity_score = distances[0][i]  # Cosine similarity
        print(f"Match {i+1}: {file_path} (Cosine Similarity: {similarity_score:.4f})")
        top_file_paths.append(file_path)

    return top_file_paths


# Main Logic
if __name__ == "__main__":
    # Predefined timestamps from transcription
    timestamps = {
        'rain': [(1.26, 1.52), (17.84, 18.12), (25.88, 26.12)],
        'forest': [(2.0, 2.32), (15.4, 15.72), (32.7, 33.0)],
        'adventure': [(33.78, 34.38)]
    }

    # Load narration
    narration = AudioSegment.from_file("narration.mp3")

    # Process each timestamp and overlay corresponding sound
    final_audio = narration

    for word, time_ranges in timestamps.items():
        # Search for the best matching audio file for the word
        matching_files = search_audio_by_text(word, k=1)

        if not matching_files:
            print(f"No matching audio file found for '{word}'")
            continue

        # Load the best matching sound file
        sound_effect = AudioSegment.from_file(matching_files[0])

        # Overlay sound effect at each timestamp
        for start, end in time_ranges:
            insert_position = int(start * 1000)  # Convert seconds to ms
            print(f"For word '{word}', inserting audio file '{matching_files[0]}' at {insert_position}ms")
            final_audio = final_audio.overlay(sound_effect, position=insert_position)

    # Export final audio
    output_file = "final_output.mp3"
    final_audio.export(output_file, format="mp3")
    print(f"Final audio saved as {output_file}")

    # Play final audio
    play(final_audio)
