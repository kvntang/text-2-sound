import torch
from transformers import ClapProcessor, ClapModel

if __name__ == "__main__":

    # Load the audio embeddings
    # Load audio embeddings with safe flag
    audio_embeddings = torch.load("audio_embeddings.pt", weights_only=True)

    # Load CLAP model and processor
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")

    # Define the text prompt
    text_prompt = ["sound of people whistling", "sound of people talking"]

    # Generate the text embeddings
    text_inputs = processor(text=text_prompt, return_tensors="pt", padding=True)

    # Extract text embeddings using the same CLAP model
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)

    # Calculate cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        audio_embeddings.unsqueeze(1),  # Expand dimensions for broadcasting
        text_embeddings.unsqueeze(0),
        dim=-1
    )

    # Display results
    for i, audio_embedding in enumerate(audio_embeddings):
        print(f"Audio File {i+1}:")
        for j, prompt in enumerate(text_prompt):
            print(f"  Similarity with '{prompt}': {cosine_similarity[i, j].item():.4f}")
