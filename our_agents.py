import autogen, asyncio
from datetime import datetime
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

def agent_community_discussion(topic):
    # Run the chat (synchronous in your AutoGen version)
    chat_manager.initiate_chat(
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

  Human context (clarification):
- The human values solitude, but not isolation.
- Regular, meaningful time with close family (spouse, parents, child, grandchildren) is grounding and welcome.
- Solitude is defined as freedom from performative, ambient, or obligatory social interaction — not from loved humans.
- Social energy is conserved for chosen relationships, not casual or default socializing (e.g., neighbors, social media).

Scope constraint:
- This conversation is ONLY about one human’s morning experience.
- Do not propose systems, programs, communities, frameworks, or organizational practices.
- If discussion drifts toward abstract groups or general populations, redirect back to the individual human.

Conversation flow constraint:
- The Skeptic must speak before the Synthesizer.
- The Synthesizer must have the final word and close the session by integrating all perspectives into a calm, grounded stance.

Conversation phase rule:
- Open exploration is encouraged early.
- Once the Synthesizer begins a closing synthesis, no new critiques or challenges should be introduced.
- The Synthesizer’s closing integration marks the end of the session.

Stopping rule:
- Once a coherent stance has been articulated, STOP.
- Do not generate “next steps,” implementation plans, or follow-on questions.
""",
        turns=20,
    )

    # ---- TRANSCRIPT SAVE (this is the new part) ----
    os.makedirs("field_notes", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = f"field_notes/{ts}_transcript.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for m in group_chat.messages:
            speaker = m.get("name") or m.get("role", "unknown")
            content = m.get("content", "")
            f.write(f"{speaker}:\n{content}\n\n")

    print(f"\nSaved transcript to: {out_path}")

if __name__ == "__main__":
    topic = """Designing a retirement-safe morning: how a human + agent community can replace the 
    corporate treadmill without becoming a new one.
    Human context:
    - The human prefers solitude and quiet.
    - Strongly dislikes social media and performative sharing.
    - Is not a “joiner” and resists group obligations.
    - Finds meaning in reflective conversation, reading, thinking, and time with dogs.
    - Wants mornings that feel spacious, non-demanding, and non-optimizing.
    - “Mornings with Iris” are about mental companionship, not social engagement.
    Constraint:
    Do not assume group participation, community rituals, or social engagement unless explicitly 
    justified for this specific human.
    """
    agent_community_discussion(topic)

