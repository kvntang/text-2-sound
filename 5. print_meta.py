import faiss
import torch

# Load the FAISS index and metadata safely
index = faiss.read_index("audio_index.faiss")

# Fix the FutureWarning by specifying `weights_only=False`
audio_metadata = torch.load("audio_metadata.pt", weights_only=True)


# Display FAISS index details
print(f"Loaded {len(audio_metadata)} audio files from the index.")
print("FAISS Index Details:")
print(f"Index Type: {type(index)}")
print(f"Index Dimension (Embedding Size): {index.d}")
print(f"Number of Stored Vectors: {index.ntotal}")

