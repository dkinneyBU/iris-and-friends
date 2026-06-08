import random
import numpy as np
from collections import Counter, deque

N_AGENTS = 50
STEPS = 500
NOISE = 0.05

# Blackboard parameters
BB_SIZE = 32              # number of slots on the blackboard
WRITE_PROB = 0.30         # chance an agent writes each step
ERASE_PROB = 0.02         # slow decay / perturbation
WINDOW = 25               # how far back to look for "norm" on the board

class Blackboard:
    def __init__(self, size):
        self.slots = [random.choice([0, 1]) for _ in range(size)]
        self.history = deque(maxlen=WINDOW)

    def step(self):
        # decay / perturbation
        for i in range(len(self.slots)):
            if random.random() < ERASE_PROB:
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
        self.behavior = random.choice([0, 1])
        self.coupling = 1.0


    def update(self, bb: Blackboard):
        # Read: align to current blackboard majority (with noise)
        if random.random() > NOISE:
            self.behavior = bb.majority()

        # Write: sometimes imprint my behavior onto a random slot
        if random.random() < (WRITE_PROB * self.coupling):

            idx = random.randrange(len(bb.slots))
            bb.slots[idx] = self.behavior

        compatible = (self.behavior == bb.majority())
        # coupling drifts down when incompatible, recovers when compatible
        if compatible:
            self.coupling = min(1.0, self.coupling + 0.05)
        else:
            self.coupling = max(0.0, self.coupling - 0.10)


def entropy_from_bits(bits):
    c = Counter(bits)
    total = len(bits)
    probs = [v / total for v in c.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def run():
    agents = [Agent() for _ in range(N_AGENTS)]
    bb = Blackboard(BB_SIZE)

    ent_agents = []
    ent_bb = []
    avg_coupling = []


    for _ in range(STEPS):
        for a in agents:
            a.update(bb)
        bb.step()

        ent_agents.append(entropy_from_bits([a.behavior for a in agents]))
        ent_bb.append(entropy_from_bits(bb.slots))
        avg_coupling.append(sum(a.coupling for a in agents) / N_AGENTS)

    print("Final agent entropy:", ent_agents[-1])
    print("Final blackboard entropy:", ent_bb[-1])
    print("Final average coupling:", avg_coupling[-1])

if __name__ == "__main__":
    run()
