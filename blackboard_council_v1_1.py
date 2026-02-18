import autogen
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file or environment variables.")

BASE_LLM_CONFIG = {
    "model": "mistral",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",   # dummy
    "temperature": 0.7,
    "price": [0, 0],  # suppress pricing warning
}

STYLE_LAW = """
Style constraints:
- No politeness or social filler. Do NOT say: thank you, please, great question, happy to help, sorry, alright, let's, as an AI, I appreciate, etc.
- No compliments. No apologies. No rhetorical framing.
- Output only content that advances the task.
- Prefer short, declarative sentences.
If you violate the style constraints, immediately rewrite without the banned content.
""".strip()

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
    system_message=(
        "No politeness. No filler. No compliments. No thanks. No apologies.\n"
        "Output ONLY the minimal V1.1 update. Ignore any mention of rewards, policy optimization, fairness, evaluation, applications, goals, performance, or decision-making.\n"
        "Do NOT introduce new mechanisms.\n"
        "Output MUST be EXACTLY these 5 numbered sections, 1–2 sentences each:\n"
        "1) How norms arise\n"
        "2) What persists\n"
        "3) How violations matter\n"
        "4) What we log\n"
        "5) What we leave unspecified\n"
        "No other text."
    ),
    temperature=0.2
)

user_proxy = autogen.UserProxyAgent(
    "User",
    llm_config=BASE_LLM_CONFIG,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,  # <-- change this
    system_message="You initiate the session and do not contribute analysis."
)

# group_chat = autogen.GroupChat(
#     agents=[user_proxy, ontologist, eliminativist, engineer, skeptic, synthesizer],
#     max_round=12,
#     messages=[],
# )
group_chat = autogen.GroupChat(
    agents=[skeptic, synthesizer],
    max_round=2,
    messages=[],
    speaker_selection_method="round_robin",
)


chat_manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=BASE_LLM_CONFIG)

def run_phase_b(mission: str):
    chat_manager.initiate_chat(
        skeptic,
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
- If reinforcement learning, reward functions, policy optimization, fairness frameworks, or large-scale evaluation infrastructure are proposed, the Adversary must explicitly reject them unless they are strictly necessary for norm persistence.

End State:
Return only:
1) a brief description of how norms arise,
2) what persists,
3) how violations matter,
4) what we log,
5) what we deliberately leave unspecified.

""",
        turns=2,
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

