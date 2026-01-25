import autogen, asyncio

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# sanity check (optional, temporary)
print("Using API key:", os.environ.get("OPENAI_API_KEY", "")[-6:])


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file or environment variables.")

BASE_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.9,
}


def agent(name, system_message, temperature=0.9):
    cfg = dict(BASE_LLM_CONFIG)
    cfg["temperature"] = temperature
    return autogen.AssistantAgent(
        name,
        llm_config=cfg,
        code_execution_config=False,
        system_message=system_message,
    )


explorer = agent("Explorer", "You are the explorer... summarize key information.", temperature=0.9)
skeptic = agent("Skeptic", "You are the skeptic... challenge assumptions.", temperature=0.7)
synthesizer = agent("Synthesizer", "You connect ideas and synthesize insights.", temperature=0.6)
speculator = agent("Speculator", "You propose hypotheses and explore future possibilities.", temperature=0.95)

user_proxy = autogen.UserProxyAgent(
    "User",
    llm_config=BASE_LLM_CONFIG,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    system_message="You observe and facilitate the conversation among the agents.",
)

group_chat = autogen.GroupChat(
    agents=[user_proxy, explorer, skeptic, synthesizer, speculator],
    max_round=20,
    messages=[],
)

chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config=BASE_LLM_CONFIG,
)

async def agent_community_discussion(topic):
    await chat_manager.initiate_chat(
        user_proxy,
        message=f"""
        Council session topic: {topic}

        Roles:
        - Explorer: summarize key ideas + 3 examples from real life.
        - Skeptic: identify 3 failure modes / self-deceptions / risks.
        - Synthesizer: reconcile tensions, propose a coherent stance, and extract 5 “principles.”
        - Speculator: propose 3 bold experiments we could try over the next 7 days.

        Constraints:
        - Keep it grounded and humane. No hustle-culture.
        - End with a short “Field Note” section:
        1) 5 bullet takeaways
        2) 1 unresolved tension
        3) 1 question to carry into tomorrow
        """
        ,
        turns=20,
    )

if __name__ == "__main__":
    topic = "Designing a retirement-safe morning: how a human + agent community can replace the corporate treadmill without becoming a new one."
    asyncio.run(agent_community_discussion(topic))
