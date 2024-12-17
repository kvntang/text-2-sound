import torch
from transformers import ClapProcessor, ClapModel
import librosa

# Function to get audio embeddings
def get_audio_embedding(file_path):
    try:
        # Load audio with librosa
        audio, sample_rate = librosa.load(file_path, sr=48000, mono=True)

        # Correct audio processing
        inputs = processor(audios=audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # Generate embeddings
        with torch.no_grad():
            embeddings = model.get_audio_features(**inputs)

        return embeddings
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Load processor and model
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused", force_download=True)
    model = ClapModel.from_pretrained("laion/clap-htsat-fused", force_download=True)

    # Define audio files
    audio_file_paths = ["hum.m4a", "meow.m4a", "whistle.m4a"]

    # Extract embeddings
    audio_embeddings = []

    for file_path in audio_file_paths:
        embedding = get_audio_embedding(file_path)
        if embedding is not None:
            audio_embeddings.append(embedding)

    # Save embeddings if not empty
    if audio_embeddings:
        audio_embeddings = torch.cat(audio_embeddings)
        torch.save(audio_embeddings, "audio_embeddings.pt")
        print("Audio embeddings saved successfully.")
