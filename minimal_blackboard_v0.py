import random
import numpy as np
from collections import Counter

N_AGENTS = 50
STEPS = 500
NEIGHBOR_RADIUS = 5
NOISE = 0.05

class Agent:
    def __init__(self):
        self.behavior = random.choice([0, 1])

    def update(self, population):
        neighbors = random.sample(population, NEIGHBOR_RADIUS)
        majority = Counter(a.behavior for a in neighbors).most_common(1)[0][0]

        if random.random() > NOISE:
            self.behavior = majority

def entropy(distribution):
    total = sum(distribution.values())
    probs = [v / total for v in distribution.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def run():
    agents = [Agent() for _ in range(N_AGENTS)]
    history = []

    for _ in range(STEPS):
        for agent in agents:
            agent.update(agents)

        dist = Counter(a.behavior for a in agents)
        history.append(entropy(dist))

    return history

if __name__ == "__main__":
    h = run()
    print("Final entropy:", h[-1])
