import os
from dotenv import load_dotenv
from google import genai

# Load the API key from your .env file
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print("Fetching available models for this specific API Key...\n")
try:
    # Ask Google's servers for the allowed models
    for m in client.models.list():
        # Filter to only show the relevant vision/flash/pro models so the terminal isn't spammed
        if "flash" in m.name or "pro" in m.name or "vision" in m.name:
            print(f"Allowed Model: {m.name}")
except Exception as e:
    print(f"Error fetching models: {e}")