import os
import autogen
from typing import Dict, List
from agents.transcript_to_meaning_generator import TRANSCRIPT_TO_MEANING_GENERATOR
# from agents.transcript_to_meaning_generator import TRANSCRIPT_TO_MEANING_GENERATOR_INSTRUCTION
from config.settings import CONFIG_LIST, LLM_CONFIG
import google.generativeai as genai

def create_agent_config(name: str, **kwargs) -> Dict:
    return {
        "name": name,
        "llm_config": LLM_CONFIG,
        **kwargs # for any other potential configs
    }

AGENT_CONFIGS = {
    TRANSCRIPT_TO_MEANING_GENERATOR["name"]: {
        **TRANSCRIPT_TO_MEANING_GENERATOR,
        "llm_config": LLM_CONFIG,
    }
}

def create_agents() -> Dict[str, autogen.AssistantAgent]:
    return {
        name: autogen.AssistantAgent(**config) 
        for name, config in AGENT_CONFIGS.items()
    }

def create_groupchat(agents: List[autogen.ConversableAgent]) -> autogen.GroupChat:
    return autogen.GroupChat(agents=agents, messages=[], max_round=10, allow_repeat_speaker=False)

def create_manager(groupchat: autogen.GroupChat) -> autogen.GroupChatManager:
    return autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": CONFIG_LIST})