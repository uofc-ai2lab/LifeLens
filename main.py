import os, json, sys, time
from typing import List, Dict
import autogen
import google.generativeai as genai
from autogen.tools import Tool
from skills.save_text_output import save_text_output, TextOutputFields
from config.settings import USER_PROXY_CONFIG
from config.agent_config import create_agents, create_groupchat, create_manager
import time

# Start time
time_start = time.time()
time_end = time_start

# ----------------------------
# Config
# ----------------------------
TRANSCRIPT_DIR = "input"
TRANSCRIPT_FILES = ["transcript1.json", "transcript2.json"]

# ----------------------------
# Load transcripts
# ----------------------------
def load_transcripts(transcript_dir: str, transcript_files: List[str]) -> Dict[str, dict]:
    transcripts = {}

    if not os.path.isdir(transcript_dir):
        raise FileNotFoundError(f"Directory '{transcript_dir}' does not exist.")

    for transcript_file in transcript_files:
        file_path = os.path.join(transcript_dir, transcript_file)
        if not transcript_file.lower().endswith(".json"):
            print(f"Skipping non-JSON file '{transcript_file}'")
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    transcripts[transcript_file] = json.loads(content)
                else:
                    print(f"Warning: File '{transcript_file}' is empty.")
        except Exception as e:
            print(f"Error reading '{transcript_file}': {e}")
    return transcripts

# ----------------------------
# User Proxy Message Handler
# ----------------------------
def user_proxy_receive(proxy, message: dict, sender: autogen.Agent, filename: str) -> None:
    """
    Handles incoming messages from the LLM agent,
    saves them to a text file with timestamp.
    """
    
    content = message.get("content", "")
    if not content.strip():
        return
    
    # The tool is already registered, so we can access its function.
    # Assuming `save_text_output` is available directly here or imported:
    output_data = TextOutputFields(filename=filename, content=content)
    save_text_output(output_data) # Direct call to the function

    if "TERMINATE" in content:
        print(f"All outputs saved to '{filename}'. Terminating.")
        time_end = time.time()
        total_time = time_end - time_start
        sys.exit(0)

# ----------------------------
# Generate data
# ----------------------------
def generate_data(transcripts: Dict[str, dict], requirements: str, filename: str):
    user_proxy = autogen.UserProxyAgent(**USER_PROXY_CONFIG)
    
    # --- Begin tool registration ---
    tool_save_text_output = Tool(func_or_tool=save_text_output, name="save_text_output",
                                 description="Save structured transcript output to text file")
    user_proxy.register_for_llm(name="save_text_output", description=tool_save_text_output.description)(tool_save_text_output)
    user_proxy.register_for_execution(name="save_text_output")(tool_save_text_output)
    # --- End tool registration ---
    
    user_proxy.receive = lambda message, sender, request_reply=False, silent=False: user_proxy_receive(
        user_proxy, message, sender, filename
    )
    
    agents = create_agents()
    all_agents = list(agents.values()) + [user_proxy]
    groupchat = create_groupchat(all_agents)
    manager = create_manager(groupchat)

    # Prepare message for the LLM
    message = f"Transcripts(s): {json.dumps(transcripts, indent=4)}\n\nRequirements: {requirements}"
    user_proxy.initiate_chat(manager, message=message)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    transcripts = load_transcripts(TRANSCRIPT_DIR, TRANSCRIPT_FILES)
    requirements = "Extract structured medical context from each transcript, preserving timestamps."
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.txt"
    generate_data(transcripts, requirements, filename)
