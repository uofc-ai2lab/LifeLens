import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import autogen

# No need to pass the key, it is loaded from the environment variable
# genai.configure()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
    
# ----------------------------
# LLM Configuration (Google Gemini)
# ----------------------------

# Autogen needs the config to be in a specific format to work with Gemini
# Note: As of AutoGen 0.2, Gemini doesn't use a 'system_message' field.
# The system instruction will be added to the initial user message instead.
# `api_type` should be 'google', and the API key can be set as an environment variable
# named `GOOGLE_GEMINI_API_KEY` or passed directly via `api_key`.
CONFIG_LIST = [
    {
        "model": "models/gemini-2.5-flash",
        "api_type": "google",
        "api_key":api_key
    },
]

LLM_CONFIG = {
    "config_list": CONFIG_LIST,
    "temperature": 0.5,
    "cache_seed": None, # or a number for caching
}

# ----------------------------
# User Proxy Configuration
# ----------------------------
USER_PROXY_CONFIG = {
    "name": "user_proxy",
    "system_message": 
    """
        You are the interface between the user and the agent system. Your role is to:
            1. Interpret user requirements and initiate the transcript to meaning process
            2. Monitor the conversation and identify when the final output is ready
            3. Extract the JSON output for each table and save it to separate files
            4. Terminate the conversation when all outputs have been saved
    """,
    "code_execution_config": {"work_dir": "output"},
    "human_input_mode": "NEVER",               # NEVER=automatic, ALWAYS=ask every step
    "max_consecutive_auto_reply": 10,
    "llm_config": LLM_CONFIG,
    "is_termination_msg": lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
}
