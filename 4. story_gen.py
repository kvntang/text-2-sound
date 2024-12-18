import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")

# Replace with your actual API key
API_KEY = "your_openai_api_key"

# Define the API endpoint
API_URL = "https://api.openai.com/v1/chat/completions"

# Define the request headers
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Define the request payload
payload = {
    "model": "gpt-3.5-turbo",  # Or "gpt-4" if available
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
}

# Send the POST request
response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

# Handle the response
if response.status_code == 200:
    result = response.json()
    message = result['choices'][0]['message']['content']
    print("ChatGPT Response:", message)
else:
    print("Failed to get response:", response.status_code, response.text)
