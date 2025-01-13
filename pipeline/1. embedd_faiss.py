import torch
from transformers import ClapProcessor, ClapModel
import faiss
import librosa
import numpy as np
import os

if __name__ == "__main__":
    # Initialize CLAP processor and model
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")

    # Define audio folder path
    audio_folder = "sound_effects/sounds"

    # Load audio file paths dynamically
    audio_file_paths = [
        os.path.join(audio_folder, file) for file in os.listdir(audio_folder)
        if file.lower().endswith(('.m4a', '.wav', '.mp3'))
    ]

    # Storage for embeddings and metadata
    audio_embeddings = []
    audio_metadata = []

    # Extract and store audio embeddings
    for file_path in audio_file_paths:
        try:
            print(f"Processing {file_path}...")
            audio, sr = librosa.load(file_path, sr=48000, mono=True)
            inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                embedding = model.get_audio_features(**inputs)
                audio_embeddings.append(embedding.squeeze().numpy())
                audio_metadata.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Check if embeddings were created
    if audio_embeddings:
        # Convert list to numpy array
        audio_embeddings = np.array(audio_embeddings)

        # Create FAISS index
        embedding_dim = audio_embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)

        # Normalize and add embeddings
        faiss.normalize_L2(audio_embeddings)  # Normalize embeddings
        index.add(audio_embeddings)

        # Save index and metadata
        faiss.write_index(index, "audio_index.faiss")
        torch.save(audio_metadata, "audio_metadata.pt")
        print(f"Saved FAISS index and metadata for {len(audio_metadata)} audio files.")
    else:
        print("No audio files processed successfully.")
