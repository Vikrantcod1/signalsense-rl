import numpy as np
from agents.discretize import N_LEVELS


class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9985, seed=0):
        # state = (ns_level, ew_level, phase, yellow), 2 actions
        self.q = np.zeros((N_LEVELS, N_LEVELS, 2, 2, 2))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        return int(np.argmax(self.q[s]))

    def update(self, s, a, r, s2):
        # no terminal masking: episodes end on a time limit, not a real terminal state
        target = r + self.gamma * np.max(self.q[s2])
        self.q[s + (a,)] += self.alpha * (target - self.q[s + (a,)])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)