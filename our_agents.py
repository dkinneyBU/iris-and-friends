# %%
import asyncio  
import autogen
import os

# Start logging
logging_session_id = autogen.runtime_logging.start(config={"society": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

# %%
# Define the OpenAI-compatible LLM backend
# config list" currently for reference only
# config_list = [
#     {
#         "model": "gpt-4",  # Use 'gpt-4' or 'gpt-3.5-turbo'
#         "temperature": 0.9,
#         "cache_seed": 42,
#         "api_key":  os.environ.get("OPENAI_API_KEY"),  # Replace this with your actual key
#     }ls -la 
# ]
llm_config = {
    "model": "gpt-4",  # Use "gpt-3.5-turbo" if you prefer a cheaper option
    "api_key": os.environ.get("OPENAI_API_KEY"),  # Replace with your actual OpenAI API key
}

# %%
# Define the agents with LLM backend
explorer = autogen.AssistantAgent("Explorer", 
    llm_config={"model": "gpt-4o", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")},
    code_execution_config=False,
    system_message="You are the explorer and you gather and summarize information from various sources.")
skeptic = autogen.AssistantAgent("Skeptic", 
    llm_config={"model": "gpt-4o", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")},
    code_execution_config=False,
    system_message="You are the skeptic and you challenge assumptions and look for inconsistencies in information.")
synthesizer = autogen.AssistantAgent("Synthesizer", 
    llm_config={"model": "gpt-4o", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")},
    code_execution_config=False,
    system_message="You are the synthesizer and you Connect ideas, identify patterns, and synthesize insights.")
speculator = autogen.AssistantAgent("Speculator", 
    llm_config={"model": "gpt-4o", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")},
    code_execution_config=False,
    system_message="You are the speculator and you propose hypotheses and explore future possibilities.")

# %%
# Define a user proxy
user_proxy = autogen.UserProxyAgent("User", 
    llm_config={"model": "gpt-4o", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    system_message="You are the user and you observe and facilitate the conversation among the agents.")


# %%
# Create a group chat for all agents
group_chat = autogen.GroupChat(agents=[user_proxy, explorer, skeptic, synthesizer, speculator], 
    max_round=20, 
    messages=[])
chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config  # <-- Fix: Set LLM config for speaker selection
)

# %%
# Define the agent workflow
async def agent_community_discussion(topic):
    """Initiate a structured discussion among agents."""
    await chat_manager.initiate_chat(
        user_proxy,  # The user starts the discussion
        message=f"Let's analyze the topic: {topic}. Explorer, please start by summarizing key information.",
        turns=20  # Increase this to allow longer discussions
    )

# %%
# Run the function safely
if __name__ == "__main__":
    topic = "The societal impacts of AI"
    # Use asyncio.run(...) if you are running this script as a standalone script.
    asyncio.run(agent_community_discussion(topic))



