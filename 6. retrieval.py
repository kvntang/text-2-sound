import torch
from transformers import ClapProcessor, ClapModel
import faiss
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pydub import AudioSegment
from pydub.playback import play


def search_audio_by_text(query_text, k=3):
    """
    Search the FAISS audio index using a text query and display results.

    Args:
        query_text (str): The text query to search for.
        top_k (int): Number of top matches to retrieve.
    """

    # Load the FAISS index and metadata
    index = faiss.read_index("audio_index.faiss")
    audio_metadata = torch.load("audio_metadata.pt", weights_only=True)

    # Reload CLAP model and processor
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


    # Display results with true cosine similarity
    print(f"\nSearch Results for: '{query_text}'")
    top_file_paths = []

    for i, idx in enumerate(indices[0]):
        file_path = audio_metadata[idx]
        similarity_score = distances[0][i]  # Cosine similarity directly from FAISS
        print(f"Match {i+1}: {file_path} (Cosine Similarity: {similarity_score:.4f})")
        top_file_paths.append(file_path)
    
    return top_file_paths





# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    search_text = "a person whistling"

    # Retrieve top audio files
    top_files = search_audio_by_text(search_text)

    # Play the top files
    for file_path in top_files:
        try:
            audio = AudioSegment.from_file(file_path)
            print(f"Playing {file_path}...")
            play(audio)
        except Exception as e:
            print(f"Error playing {file_path}: {e}")