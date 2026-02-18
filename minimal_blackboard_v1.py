import random
import numpy as np
from collections import Counter, deque

N_AGENTS = 50
STEPS = 500
NOISE = 0.05

# Blackboard parameters
BB_SIZE = 32              # number of slots on the blackboard
WRITE_PROB = 0.30         # chance an agent writes each step
WINDOW = 25               # how far back to look for "norm" on the board


class Blackboard:
    def __init__(self, size, erase_prob):
        self.erase_prob = erase_prob
        self.slots = [random.choice([0, 1]) for _ in range(size)]
        self.history = deque(maxlen=WINDOW)

    def step(self):
        # decay / perturbation
        for i in range(len(self.slots)):
            if random.random() < self.erase_prob:
                self.slots[i] = random.choice([0, 1])
        self.history.append(self.snapshot())

    def snapshot(self):
        return list(self.slots)

    def majority(self):
        # Majority over recent snapshots (temporal persistence)
        if not self.history:
            bits = self.slots
        else:
            bits = [b for snap in self.history for b in snap]
        c = Counter(bits)
        return c.most_common(1)[0][0]


class Agent:
    def __init__(self):
        self.behavior_A = random.choice([0, 1])
        self.behavior_B = random.choice([0, 1])
        self.coupling = 1.0

    def update(self, bb: Blackboard, ctx: str):
        # Pick the context-specific behavior
        behavior = self.behavior_A if ctx == "A" else self.behavior_B

        # Read: align to blackboard majority (with noise)
        maj = bb.majority()
        if random.random() > NOISE:
            behavior = maj

        # Compatibility affects coupling (constraint, not reward)
        compatible = (behavior == maj)
        if compatible:
            self.coupling = min(1.0, self.coupling + 0.05)
        else:
            self.coupling = max(0.0, self.coupling - 0.10)

        # Write: imprint behavior with strength scaled by coupling
        if random.random() < (WRITE_PROB * self.coupling):
            idx = random.randrange(len(bb.slots))
            bb.slots[idx] = behavior

        # Store back into the correct context slot
        if ctx == "A":
            self.behavior_A = behavior
        else:
            self.behavior_B = behavior


def entropy_from_bits(bits):
    c = Counter(bits)
    total = len(bits)
    probs = [v / total for v in c.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def run():
    agents = [Agent() for _ in range(N_AGENTS)]

    # Two different contexts
    bbA = Blackboard(BB_SIZE, erase_prob=0.01)  # stable
    bbB = Blackboard(BB_SIZE, erase_prob=0.05)  # noisier

    ent_A_agents = []
    ent_B_agents = []
    ent_bbA = []
    ent_bbB = []
    avg_coupling = []

    for _ in range(STEPS):
        # Each agent visits exactly one context per step
        for a in agents:
            if random.random() < 0.5:
                a.update(bbA, "A")
            else:
                a.update(bbB, "B")

        # Both environments evolve
        bbA.step()
        bbB.step()

        # Logging: agent behaviors by context
        ent_A_agents.append(entropy_from_bits([a.behavior_A for a in agents]))
        ent_B_agents.append(entropy_from_bits([a.behavior_B for a in agents]))
        ent_bbA.append(entropy_from_bits(bbA.slots))
        ent_bbB.append(entropy_from_bits(bbB.slots))
        avg_coupling.append(sum(a.coupling for a in agents) / N_AGENTS)

    print("Final agent entropy (A-behaviors):", ent_A_agents[-1])
    print("Final agent entropy (B-behaviors):", ent_B_agents[-1])
    print("Final bbA entropy:", ent_bbA[-1])
    print("Final bbB entropy:", ent_bbB[-1])
    print("Final average coupling:", avg_coupling[-1])


if __name__ == "__main__":
    run()
