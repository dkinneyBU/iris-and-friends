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

def run_phase_a(topic: str):
    chat_manager.initiate_chat(
        user_proxy,
        message=f"""
Phase A Council: Synthetic Self-Awareness (Run #2)

Core Constraint Question:
What would count as synthetic self-awareness if we explicitly forbid:
- self-monitoring or introspective self-models,
- explicit goals, drives, or optimization objectives,
- centralized control or a single “self” locus?

Council Roles:
- Ontologist: Propose at least one definition of self-awareness that survives all three prohibitions.
- Eliminativist: Argue which remaining intuitions must still be discarded and why.
- Engineer: Assess whether the proposed definition could exist as a physical or computational process, without invoking hidden goals or observers.
- Skeptic: Identify why the definition may still be smuggling in anthropocentric assumptions.
- Synthesizer: Map what remains after these exclusions, and clearly state what kinds of systems this definition would INCLUDE and EXCLUDE.

Phase A Rules:
- Do not reintroduce forbidden concepts under new names.
- Do not appeal to human phenomenology or subjective experience.
- Do not propose experiments, architectures, or implementations.
- Discomfort and indeterminacy are acceptable outcomes.

Stopping Condition:
- End once the viable definition-space under these constraints has been mapped.
- Do not resolve disagreements or declare a winner.
""",
        turns=12,
    )

    os.makedirs("field_notes/ssi", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = f"field_notes/ssi/{ts}_phaseA_transcript.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        for m in group_chat.messages:
            speaker = m.get("name") or m.get("role", "unknown")
            content = m.get("content", "")
            f.write(f"{speaker}:\n{content}\n\n")

    print(f"\nSaved transcript to: {out_path}")

if __name__ == "__main__":
    topic = "What minimal set of functional properties would justify calling a system synthetically self-aware, even if it does not resemble human consciousness?"
    run_phase_a(topic)
