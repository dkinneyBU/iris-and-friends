import asyncio  
import autogen
import os

# Start logging
logging_session_id = autogen.runtime_logging.start(config={"society": "logs.db"})
print("Logging session ID: " + str(logging_session_id))

# Define the OpenAI-compatible LLM backend
config_list = [
    {
        "model": "gpt-4",  # Use 'gpt-4' or 'gpt-3.5-turbo'
        "api_key":  os.environ.get("OPENAI_API_KEY"),  # Replace this with your actual key
    }
]
llm_config = {
    "model": "gpt-4",  # Use "gpt-3.5-turbo" if you prefer a cheaper option
    "api_key": os.environ.get("OPENAI_API_KEY"),  # Replace with your actual OpenAI API key
}

# Define the agents with LLM backend
explorer = autogen.AssistantAgent("Explorer", llm_config={"config_list": config_list},
                                  description="Gathers and summarizes information from various sources.")
skeptic = autogen.AssistantAgent("Skeptic", llm_config={"config_list": config_list},
                                 description="Challenges assumptions and looks for inconsistencies in information.")
synthesizer = autogen.AssistantAgent("Synthesizer", llm_config={"config_list": config_list},
                                     description="Connects ideas, identifies patterns, and synthesizes insights.")
speculator = autogen.AssistantAgent("Speculator", llm_config={"config_list": config_list},
                                    description="Proposes hypotheses and explores future possibilities.")

# Define a user proxy
user_proxy = autogen.UserProxyAgent("User", llm_config={"config_list": config_list},
                                    human_input_mode="NEVER",
                                    max_consecutive_auto_reply=100,
                                    description="Observes and facilitates the conversation among the agents.")

# Create a group chat for all agents
group_chat = autogen.GroupChat(agents=[explorer, skeptic, synthesizer, speculator], messages=[])
chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config  # <-- Fix: Set LLM config for speaker selection
)

# Define the agent workflow
async def agent_community_discussion(topic):
    """Initiate a structured discussion among agents."""
    await chat_manager.initiate_chat(
        explorer,  # The user starts the discussion
        message=f"Let's analyze the topic: {topic}. Explorer, please start by summarizing key information.",
        max_round=20  # Increase this to allow longer discussions
    )

# Run the function safely
if __name__ == "__main__":
    topic = "The societal impacts of AI"
    asyncio.run(agent_community_discussion(topic))  # Run the async function properly
