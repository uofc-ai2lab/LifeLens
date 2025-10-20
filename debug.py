import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# No need to pass the key, it is loaded from the environment variable
# genai.configure()

'''
To verify the exact model names available to you. 
Google provides an API to list available models.
Run this file to see the list of available models.
'''
API_KEY = os.getenv("GOOGLE_API_KEY") 

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=API_KEY) # Explicitly configure genai with the key

    print("Attempting to list available Gemini models...")
    try:
        for m in genai.list_models():
            # Filter for generative models and check for generateContent support
            if "generateContent" in m.supported_generation_methods:
                print(f"  - Model: {m.name}, Supported: {m.supported_generation_methods}")
    except Exception as e:
        print(f"An error occurred while listing models: {e}")
        
        