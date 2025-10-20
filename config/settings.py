import os
import autogen
from dotenv import load_dotenv
load_dotenv()

# option 1: transcript to meaning
# option 2: deidentify data - then send to saba

# ----------------------------
# LLM Configuration (DeepSeek)
# ----------------------------

CONFIG_LIST = [
    {
        "model": "deepseek-r1-distill-llama-8b",  # local DeepSeek model
        "base_url": "http://localhost:1234/v1",   # DeepSeek server endpoint
        "api_key": None                            # local model does not require a key
    },
]

LLM_CONFIG = {
    "timeout": 600,            # kills request after certain amount of time
    "seed": 42,                # for caching / deterministic results
    "config_list": CONFIG_LIST,
    "temperature": 0.1         # low = deterministic, higher = more creative
}

LLM_CONFIG_HIGH = {
    "timeout": 600,
    "seed": 42,
    "config_list": CONFIG_LIST,
    "temperature": 1.0         # more creative responses
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
