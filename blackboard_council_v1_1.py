import autogen
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file or environment variables.")

BASE_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.9,
    "api_key": OPENAI_API_KEY,
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

# Phase A Council roles (replace Explorer/Speculator for this script)
ontologist = agent(
    "Ontologist",
    "Define synthetic self-awareness in non-biological terms. Use operational/functional definitions, not feelings.",
    temperature=0.8,
)
eliminativist = agent(
    "Eliminativist",
    "Identify which human concepts (qualia, ego, intuition, etc.) should be discarded when discussing synthetic minds. Be ruthless and clear.",
    temperature=0.9,
)
engineer = agent(
    "Engineer",
    "Assess which proposed criteria could plausibly exist in a machine. Flag magic/hand-waving.",
    temperature=0.6,
)
skeptic = agent(
    "Skeptic",
    "Challenge definitions as circular, insufficient, or anthropocentric. Pressure-test assumptions.",
    temperature=0.8,
)
synthesizer = agent(
    "Synthesizer",
    "Map the space of definitions and disagreements. Summarize fault lines without forcing convergence or closure.",
    temperature=0.6,
)

user_proxy = autogen.UserProxyAgent(
    "User",
    llm_config=BASE_LLM_CONFIG,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    system_message="You initiate the council and facilitate turn-taking without adding new content.",
)

group_chat = autogen.GroupChat(
    agents=[user_proxy, ontologist, eliminativist, engineer, skeptic, synthesizer],
    max_round=12,
    messages=[],
)

chat_manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=BASE_LLM_CONFIG)

def run_phase_b(mission: str):
    chat_manager.initiate_chat(
        user_proxy,
        message=f"""
Phase B Council: Blackboard Ecology V1.1 — Norm Formation

Mission:
Refine the V1 blackboard ecology so that it can support the emergence and persistence of internally generated conventions (“norms”) WITHOUT:
- explicit goals or rewards,
- centralized control,
- predefined rule-learning,
- self-models or introspection.

Clarification:
A “norm” here means a pattern, convention, or constraint that:
- was not explicitly encoded,
- persists across time,
- constrains future behavior,
- and whose violation has systemic consequences.

Roles:
- Ecologist: Describe what minimal environmental conditions allow norms to arise and persist.
- Mechanist: Explain how norms could stabilize as structural regularities rather than explicit rules.
- Adversary: Identify how apparent norms could collapse into noise, overfitting, or illusion.
- Observer: Specify how we could detect norms empirically without naming or labeling them.
- Synthesizer: Produce a concise V1.1 update that adds norm-formation capacity to the blackboard ecology while keeping it minimal and implementable.

Constraints:
- Norms must arise from interaction, not instruction.
- Do not introduce scoring, rewards, or fitness functions.
- Avoid anthropomorphic language (no “beliefs,” “intentions,” or “understanding”).
- Keep the update compatible with a few-hundred-line Python implementation.

End State:
Return only:
1) a brief description of how norms arise,
2) what persists,
3) how violations matter,
4) what we log,
5) what we deliberately leave unspecified.

""",
        turns=10,
    )

    os.makedirs("field_notes/ssi", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = f"field_notes/ssi/{ts}_phaseB_transcript.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for m in group_chat.messages:
            speaker = m.get("name") or m.get("role", "unknown")
            content = m.get("content", "")
            f.write(f"{speaker}:\n{content}\n\n")

    print(f"\nSaved transcript to: {out_path}")

if __name__ == "__main__":
    mission = (
        "Design the smallest possible computational system in which persistent, adaptive, "
        "self-maintaining structure could emerge WITHOUT centralized control, explicit goals/"
        "reward functions, or self-models/introspection."
    )
    run_phase_b(mission)

