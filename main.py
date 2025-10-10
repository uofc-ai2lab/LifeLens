import autogen, json, sys, time, os # type: ignore
from utils.agent_utils import create_agents, create_groupchat, create_manager
from config.settings import USER_PROXY_CONFIG
from skills.save_text_output import save_text_output, TextOutputFields
from autogen.tools import Tool

TRANSCRIPT_DIR = "input"  # Directory where the AVSC files are located
TRANSCRIPT_FILES = ["transcript1.json","transcript2.json"]


def user_proxy_receive(self, message: dict, sender: autogen.Agent, filename: str) -> None:
    user_proxy = autogen.UserProxyAgent(**USER_PROXY_CONFIG)

    # For TEXT OUTPUT skill
    input_data_text = TextOutputFields(filename=filename, content=message["content"])
    tool_text = Tool(func_or_tool=save_text_output, name="save_text_output",  description="A file writing tool for logging purposes to keep track of all the test data being generated")
    user_proxy.register_for_llm(name="save_text_output", description="A file writing tool for logging purposes to keep track of all the test data being generated")(tool_text)
    user_proxy.register_for_execution(name="save_text_output")(tool_text)
    tool_text.func(input_data_text)

    if is_final_output(message):
        print("All outputs saved. Terminating conversation.")
        sys.exit(0)
       
def is_final_output(message: dict) -> bool:
    content = message.get("content", "")
    return "TERMINATE" in content

def user_proxy_receive(self, message: dict, sender: autogen.Agent, filename: str) -> None:
    user_proxy = autogen.UserProxyAgent(**USER_PROXY_CONFIG)
    if is_final_output(message):
        print("All outputs saved. Terminating conversation.")
        sys.exit(0)

def generate_data(transcripts: dict, requirements: str, filename: str):
    user_proxy = autogen.UserProxyAgent(**USER_PROXY_CONFIG)
    user_proxy.receive = lambda message, sender, request_reply, silent: user_proxy_receive(user_proxy, message, sender, filename)
    
    agents = create_agents()
    all_agents = list(agents.values()) + [user_proxy]
    groupchat = create_groupchat(all_agents)
    manager = create_manager(groupchat)

    message = f"Transcripts(s): {json.dumps(transcripts, indent=4)}\n\nRequirements: {requirements}"
    user_proxy.initiate_chat(manager, message=message)

import os
import json
from typing import List, Dict

def load_transcripts(transcript_dir: str, transcript_files: List[str]) -> Dict[str, dict]:
    """
    Load multiple transcript JSON files from a directory.

    Args:
        transcript_dir (str): Directory containing transcript JSON files.
        transcript_files (List[str]): List of transcript filenames to load.

    Returns:
        Dict[str, dict]: Dictionary mapping filenames to their loaded JSON contents.
    """
    transcripts = {}

    # Ensure the directory exists
    if not os.path.isdir(transcript_dir):
        raise FileNotFoundError(f"Error: Directory '{transcript_dir}' does not exist.")

    for transcript_file in transcript_files:
        schema_path = os.path.join(transcript_dir, transcript_file)

        # Check file extension
        if not transcript_file.lower().endswith(".json"):
            print(f"Warning: Skipping '{transcript_file}' (not a JSON file).")
            continue

        # Load file contents
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"Warning: File '{transcript_file}' is empty — skipping.")
                    continue
                transcripts[transcript_file] = json.loads(content)

        except FileNotFoundError:
            print(f"Error: File '{transcript_file}' not found in '{transcript_dir}'.")
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON in '{transcript_file}' — {e}")
        except Exception as e:
            print(f"Unexpected error loading '{transcript_file}': {e}")

    if not transcripts:
        print("Warning: No valid transcripts loaded.")

    return transcripts



if __name__ == "__main__":
    transcripts = load_transcripts(TRANSCRIPT_DIR, TRANSCRIPT_FILES)

    requirements = "" # todp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.txt"
    generate_data(transcripts, requirements, filename)
