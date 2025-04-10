# %% Imports and logging initialization
import asyncio  
import autogen
import os
from pprint import pprint

# Start logging
logging_session_id = autogen.runtime_logging.start(config={"society": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

# %% Define the llm configuration parameters
llm_config = {
    "model": "gpt-4", 
    "temperature": 0.9,
    "cache_seed": 42,
    "api_key": os.environ.get("OPENAI_API_KEY"),
}

# %% Define the agents with LLM backend
explorer = autogen.AssistantAgent("Explorer", 
    llm_config=llm_config,
    code_execution_config=False,
    system_message="You are the explorer and you gather and summarize information from various sources.")

skeptic = autogen.AssistantAgent("Skeptic", 
    llm_config=llm_config,
    code_execution_config=False,
    system_message="You are the skeptic and you challenge assumptions and look for inconsistencies in information.")

synthesizer = autogen.AssistantAgent("Synthesizer", 
    llm_config=llm_config,
    code_execution_config=False,
    system_message="You are the synthesizer and you Connect ideas, identify patterns, and synthesize insights.")

speculator = autogen.AssistantAgent("Speculator", 
    llm_config=llm_config,
    code_execution_config=False,
    system_message="You are the speculator and you propose hypotheses and explore future possibilities.")

# %% Define a user proxy
user_proxy = autogen.UserProxyAgent("User", 
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    system_message="You are the user and you observe and facilitate the conversation among the agents.")


# %% Create a group chat for all agents
group_chat = autogen.GroupChat(agents=[user_proxy, explorer, skeptic, synthesizer, speculator], 
    max_round=20, 
    messages=[])
chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config  
)

# %% Define the agent workflow
def agent_community_discussion(topic):
    """Initiate a structured discussion among agents."""
    chat_result=chat_manager.initiate_chat(
        user_proxy,  # The user starts the discussion
        message=f"Let's analyze the topic: {topic}. Explorer, please start by summarizing key information.",
        max_turns=20
    )
    # Get the cost of the chat.
    pprint(chat_result.chat_history)
    pprint(chat_result.cost)

# %% Run the function safely
if __name__ == "__main__":
    topic = "The societal impacts of AI"
    # Use asyncio.run(...) if you are running this script as a standalone script
    # or create_task(...) if running within the editor.
    agent_community_discussion(topic)

# %% Retrieve cache by key
from autogen.cache import Cache

cache = Cache()

# Example of reading from the cache
cached_response = cache.get("cache.db-x-Cache-29-key.bin", "default_value_if_not_found")



# %%
