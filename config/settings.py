import os
import autogen
from dotenv import load_dotenv

load_dotenv()

# option 1: transcript to meaning
# option 2: deidentify data - then send to saba

api_key = os.getenv("OPENAI_API_KEY") # 'model: 'gpt-3.5-turbo
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment variables!")

CONFIG_LIST = [
    {
        "model": "gpt-3.5-turbo", # for openai: gpt-3.5-turbo
        "base_url": "https://api.openai.com/v1", # for openai: https://api.openai.com/v1
        "api_key": api_key 
    },
]

LLM_CONFIG={
    "timeout": 600,     # kills request after certain amount of time
    "seed": 42,                 # for caching
    "config_list": CONFIG_LIST,
    "temperature": 0.1           # higher temperature = more creative responses 
}

LLM_CONFIG_HIGH={
    "timeout": 600,     # kills request after certain amount of time
    "seed": 42,                 # for caching
    "config_list": CONFIG_LIST,
    "temperature": 10           # higher temperature = more creative responses 
}


USER_PROXY_CONFIG = {
    "name": "user_proxy",
    "system_message": 
    """
        You are the interface between the user and the agent system. Your role is to:
            1. Interpret user requirements and initiate the transcript to meaning process
            2. Monitor the conversation and identify when the final output is ready
            3. Extract the JSON output for each table and save it to separate files
            4. Terminate the conversation when all outputs have been saved
            5. Agents order should always be 
    """,
    "code_execution_config":{"work_dir": "output"},
    "human_input_mode": "NEVER", #whether to ask for human input severy time a message is received ALWAYS=every step, TERMINATE=when task is completed, asks for feedback
    "max_consecutive_auto_reply": 10,
    "llm_config":LLM_CONFIG,
    "is_termination_msg":lambda x: x.get("content", "").rstrip().endswith("TERMINATE") # keyword that ends task
}
# user proxy -> acts on behalf of the user, can do things automatically like executing code and responding to the assistant agent, etc.