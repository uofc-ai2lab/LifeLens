import os
import autogen
from typing import Dict, List, Optional
from agents.transcript_to_meaning_generator import TRANSCRIPT_TO_MEANING_GENERATOR
from config.settings import CONFIG_LIST, LLM_CONFIG, LLM_CONFIG_HIGH
from typing import Dict, Optional

def create_agent_config(name: str, system_message: str) -> Dict:
    return {
        "name": name,
        "llm_config": LLM_CONFIG,
        "system_message": system_message
    }

def create_agent_config_high(name: str, system_message: str) -> Dict:
    return {
        "name": name,
        "llm_config":LLM_CONFIG_HIGH,
        "system_message": system_message
    }

AGENT_CONFIGS = {
    "TranscriptToMeaningGenerator":create_agent_config(**TRANSCRIPT_TO_MEANING_GENERATOR)
}


def create_agents() -> Dict[str, autogen.AssistantAgent]:
    return {name: autogen.AssistantAgent(**config) for name, config in AGENT_CONFIGS.items()}

def create_groupchat(agents: List[autogen.ConversableAgent]) -> autogen.GroupChat:
    return autogen.GroupChat(agents=agents, messages=[], max_round=10, allow_repeat_speaker=False)

def create_manager(groupchat: autogen.GroupChat) -> autogen.GroupChatManager:
    return autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": CONFIG_LIST})