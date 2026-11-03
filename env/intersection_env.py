import gym
import numpy as np
from gym import spaces


class IntersectionEnv(gym.Env):
    """4-way intersection. Lanes: 0=N, 1=S, 2=E, 3=W.
    Phase 0 = N-S green, phase 1 = E-W green.
    Actions: 0 = keep current phase, 1 = switch phase.
    """

    def __init__(self,
                 arrival_rates=(0.3, 0.3, 0.1, 0.1),  # avg cars arriving per step
                 max_queue=20,
                 episode_len=200,
                 switch_penalty=1.0,
                 discharge=1):                         # cars leaving per green lane per step
        super().__init__()
        self.arrival_rates = np.array(arrival_rates, dtype=np.float32)
        self.max_queue = max_queue
        self.episode_len = episode_len
        self.switch_penalty = switch_penalty
        self.discharge = discharge

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=max_queue, shape=(5,), dtype=np.float32
        )
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def _obs(self):
        return np.append(self.queues, self.phase).astype(np.float32)

    def reset(self):
        self.queues = np.zeros(4, dtype=np.int32)
        self.phase = 0
        self.t = 0
        return self._obs()

    def step(self, action):
        switched = int(action == 1)
        if switched:
            self.phase = 1 - self.phase

        # cars leave on the green lanes
        green = [0, 1] if self.phase == 0 else [2, 3]
        for lane in green:
            self.queues[lane] = max(0, self.queues[lane] - self.discharge)

        # new cars arrive (Poisson), capped at max_queue
        arrivals = self.rng.poisson(self.arrival_rates)
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

        self.t += 1
        total_queue = int(self.queues.sum())
        reward = -total_queue - self.switch_penalty * switched
        done = self.t >= self.episode_len
        info = {"total_queue": total_queue, "switched": switched}
        return self._obs(), float(reward), done, info