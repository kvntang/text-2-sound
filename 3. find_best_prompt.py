import torch
from transformers import ClapProcessor, ClapModel

if __name__ == "__main__":
    # Load the audio embeddings
    audio_embeddings = torch.load("audio_embeddings.pt", weights_only=True)

    # Load CLAP model and processor
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")

    # Define candidate prompts for automatic generation
    candidate_prompts = [
    "sound of rain falling",
    "sound of thunder rumbling",
    "sound of wind blowing through trees",
    "sound of ocean waves crashing",
    "sound of a river flowing",
    "sound of leaves rustling",
    "sound of a waterfall",
    "sound of a crackling campfire",
    "sound of birds chirping",
    "sound of crickets at night",
    "sound of a dog barking",
    "sound of a cat meowing",
    "sound of a horse neighing",
    "sound of a cow mooing",
    "sound of a rooster crowing",
    "sound of a wolf howling",
    "sound of a lion roaring",
    "sound of a bird singing",
    "sound of a bee buzzing",
    "sound of an elephant trumpeting",
    "sound of people talking",
    "sound of children laughing",
    "sound of people cheering",
    "sound of people clapping",
    "sound of people shouting",
    "sound of footsteps on gravel",
    "sound of someone typing on a keyboard",
    "sound of people singing in a choir",
    "sound of someone whistling",
    "sound of a baby crying",
    "sound of a piano playing",
    "sound of a guitar strumming",
    "sound of a violin playing",
    "sound of a flute playing",
    "sound of a drum beating",
    "sound of a saxophone playing",
    "sound of an orchestra tuning up",
    "sound of an electric guitar solo",
    "sound of a church bell ringing",
    "sound of a harmonica playing",
    "sound of a car engine starting",
    "sound of a train passing by",
    "sound of an airplane flying overhead",
    "sound of a motorcycle engine revving",
    "sound of a helicopter hovering",
    "sound of a lawnmower running",
    "sound of a washing machine spinning",
    "sound of a vacuum cleaner",
    "sound of an old typewriter",
    "sound of an electric saw cutting wood",
    "sound of a busy street",
    "sound of cars honking",
    "sound of a subway train arriving",
    "sound of a police siren",
    "sound of a fire truck",
    "sound of a public announcement in a station",
    "sound of a doorbell ringing",
    "sound of a creaky door opening",
    "sound of someone knocking on a door",
    "sound of an elevator moving",
    "sound of a thunderstorm",
    "sound of strong wind gusts",
    "sound of hail hitting a roof",
    "sound of snow crunching underfoot",
    "sound of freezing rain",
    "sound of wind howling in the mountains",
    "sound of a hurricane approaching",
    "sound of raindrops hitting a window",
    "sound of a calm lake at dawn",
    "sound of a foghorn in the distance",
    "sound of water boiling",
    "sound of a kettle whistling",
    "sound of coffee brewing",
    "sound of dishes clinking",
    "sound of a microwave beeping",
    "sound of a refrigerator humming",
    "sound of a clock ticking",
    "sound of blinds being pulled down",
    "sound of pages being turned in a book",
    "sound of a pen writing on paper",
    "sound of a futuristic engine",
    "sound of a laser blast",
    "sound of a robot talking",
    "sound of a spaceship launching",
    "sound of a digital alert beep",
    "sound of a teleportation effect",
    "sound of a computer booting up",
    "sound of AI voice assistant talking",
    "sound of typing on a holographic keyboard",
    "sound of a drone flying overhead",
    "sound of glass breaking",
    "sound of wood being chopped",
    "sound of coins jingling",
    "sound of a zipper being pulled",
    "sound of a match being struck",
    "sound of fireworks exploding",
    "sound of applause in a large stadium",
    "sound of a basketball bouncing",
    "sound of a tennis ball being hit",
    "sound of a crowd roaring at a concert"
]


    # Generate text embeddings for all prompts
    text_inputs = processor(text=candidate_prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)

    # Find best matching prompts for each audio file
    for i, audio_embedding in enumerate(audio_embeddings):
        # Calculate cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(
            audio_embedding.unsqueeze(0),  # Keep audio as a single vector
            text_embeddings,
            dim=-1
        )

        # Find the best matching prompt
        best_index = torch.argmax(cosine_similarity).item()
        best_prompt = candidate_prompts[best_index]
        best_score = cosine_similarity[best_index].item()

        # Display the result
        print(f"Audio File {i+1}: Best Match -> '{best_prompt}' with Similarity Score: {best_score:.4f}")
