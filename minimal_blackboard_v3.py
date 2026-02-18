import random
import numpy as np
from collections import Counter, deque

N_AGENTS = 50
STEPS = 500
NOISE = 0.05

BB_SIZE = 32
WRITE_PROB = 0.30
WINDOW = 25

MUTATE_PROB_A = 0.005
MUTATE_PROB_B = 0.03


class Blackboard:
    def __init__(self, size, erase_prob):
        self.erase_prob = erase_prob
        self.slots = [random.choice([0, 1]) for _ in range(size)]
        self.history = deque(maxlen=WINDOW)

        # instability tracking (instantaneous majority flips)
        self.last_majority = None
        self.flip_rate = 0.0

    def snapshot(self):
        return list(self.slots)

    def instant_majority(self):
        c = Counter(self.slots)
        return c.most_common(1)[0][0]

    def majority(self):
        # Temporal majority over recent snapshots (persistence signal)
        if not self.history:
            bits = self.slots
        else:
            bits = [b for snap in self.history for b in snap]
        c = Counter(bits)
        return c.most_common(1)[0][0]

    def step(self):
        # decay / perturbation
        for i in range(len(self.slots)):
            if random.random() < self.erase_prob:
                self.slots[i] = random.choice([0, 1])

        # record history after decay
        self.history.append(self.snapshot())

        # update flip-rate based on instantaneous majority
        m = self.instant_majority()
        if self.last_majority is None:
            self.last_majority = m
        else:
            flipped = 1.0 if m != self.last_majority else 0.0
            self.flip_rate = 0.9 * self.flip_rate + 0.1 * flipped
            self.last_majority = m

    def majority_margin(self):
        c = Counter(self.slots)
        n0 = c.get(0, 0)
        n1 = c.get(1, 0)
        return abs(n0 - n1) / len(self.slots)   # 0..1


class Agent:
    def __init__(self):
        self.behavior_A = random.choice([0, 1])
        self.behavior_B = random.choice([0, 1])
        self.coupling_A = 1.0
        self.coupling_B = 1.0


    def update(self, bb: Blackboard, ctx: str):
        behavior = self.behavior_A if ctx == "A" else self.behavior_B
        coupling = self.coupling_A if ctx == "A" else self.coupling_B


        # Read: align to temporal majority (with noise)
        maj = bb.majority()
        read_prob = 0.30 if ctx == "A" else 0.10
        if random.random() < read_prob:
            behavior = maj

        # Mutate: occasional drift (creates violations without rewards)
        mut = MUTATE_PROB_A if ctx == "A" else MUTATE_PROB_B
        if random.random() < mut:
            behavior = 1 - behavior

        # Compatibility (instantaneous majority is the "current room vibe")
        inst = bb.instant_majority()
        compatible = (behavior == inst)

        # coupling drifts down when incompatible, recovers when compatible
        if compatible:
            coupling = min(1.0, coupling + 0.01)
        else:
            coupling = max(0.0, coupling - 0.20)

        # Context instability reduces influence
        coupling = max(0.0, coupling - 0.2 * bb.flip_rate)

        # Write: imprint behavior with strength scaled by coupling
        if random.random() < (WRITE_PROB * coupling):
            idx = random.randrange(len(bb.slots))
            bb.slots[idx] = behavior

        # Store back into the correct context slot
        if ctx == "A":
            self.behavior_A = behavior
            self.coupling_A = coupling
        else:
            self.behavior_B = behavior
            self.coupling_B = coupling


def entropy_from_bits(bits):
    c = Counter(bits)
    total = len(bits)
    probs = [v / total for v in c.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def run():
    agents = [Agent() for _ in range(N_AGENTS)]

    bbA = Blackboard(BB_SIZE, erase_prob=0.01)  # stable
    bbB = Blackboard(BB_SIZE, erase_prob=0.03)  # noisier

    ent_A_agents = []
    ent_B_agents = []
    ent_bbA = []
    ent_bbB = []
    avg_coupling_A = []
    avg_coupling_B = []
    majA = []
    majB = []
    marginA = []
    marginB = []

    def max_run(xs):
        best = 1
        cur = 1
        for i in range(1, len(xs)):
            if xs[i] == xs[i-1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best    

    for _ in range(STEPS):
        for a in agents:
            if random.random() < 0.5:
                a.update(bbA, "A")
            else:
                a.update(bbB, "B")

        bbA.step()
        bbB.step()

        ent_A_agents.append(entropy_from_bits([a.behavior_A for a in agents]))
        ent_B_agents.append(entropy_from_bits([a.behavior_B for a in agents]))
        ent_bbA.append(entropy_from_bits(bbA.slots))
        ent_bbB.append(entropy_from_bits(bbB.slots))
        avg_coupling_A.append(sum(a.coupling_A for a in agents) / N_AGENTS)
        avg_coupling_B.append(sum(a.coupling_B for a in agents) / N_AGENTS)
        majA.append(bbA.instant_majority())
        majB.append(bbB.instant_majority())
        marginA.append(bbA.majority_margin())
        marginB.append(bbB.majority_margin())

    print("Final agent entropy (A-behaviors):", ent_A_agents[-1])
    print("Final agent entropy (B-behaviors):", ent_B_agents[-1])
    print("Final bbA entropy:", ent_bbA[-1])
    print("Final bbB entropy:", ent_bbB[-1])
    print("Final average coupling (A):", avg_coupling_A[-1])
    print("Final average coupling (B):", avg_coupling_B[-1])

    print("Final bbA flip_rate:", bbA.flip_rate)
    print("Final bbB flip_rate:", bbB.flip_rate)

    print("Min coupling A:", min(a.coupling_A for a in agents))
    print("Min coupling B:", min(a.coupling_B for a in agents))

    print("Avg coupling A (mean):", sum(avg_coupling_A)/len(avg_coupling_A))
    print("Avg coupling B (mean):", sum(avg_coupling_B)/len(avg_coupling_B))

    print("Max majority run A:", max_run(majA))
    print("Max majority run B:", max_run(majB))    

    print("Avg margin A:", sum(marginA)/len(marginA))
    print("Avg margin B:", sum(marginB)/len(marginB))
    print("Min margin A:", min(marginA))
    print("Min margin B:", min(marginB))

    print("\nFIELD NOTE")
    print("A: avg_margin", round(sum(marginA)/len(marginA), 3),
        "min_margin", round(min(marginA), 3),
        "avg_coupling", round(sum(avg_coupling_A)/len(avg_coupling_A), 3))
    print("B: avg_margin", round(sum(marginB)/len(marginB), 3),
        "min_margin", round(min(marginB), 3),
        "avg_coupling", round(sum(avg_coupling_B)/len(avg_coupling_B), 3))



if __name__ == "__main__":
    run()
