import gym
import numpy as np
from gym import spaces


class IntersectionEnv(gym.Env):
    """4-way intersection. Lanes: 0=N, 1=S, 2=E, 3=W.
    Phase 0 = N-S green, phase 1 = E-W green.
    Actions: 0 = keep current phase, 1 = switch phase.
    After a switch there is a yellow period where no cars move.
    pattern="static": arrival_rates stay constant.
    pattern="rush":   N-S heavy -> balanced -> E-W heavy within an episode.
    """

    RUSH_RATES = [
        (0.3, 0.3, 0.1, 0.1),   # N-S heavy
        (0.2, 0.2, 0.2, 0.2),   # balanced
        (0.1, 0.1, 0.3, 0.3),   # E-W heavy
    ]

    def __init__(self,
                 arrival_rates=(0.3, 0.3, 0.1, 0.1),
                 pattern="static",
                 max_queue=20,
                 episode_len=200,
                 switch_penalty=1.0,
                 yellow_time=2,
                 discharge=1):
        super().__init__()
        self.arrival_rates = np.array(arrival_rates, dtype=np.float32)
        self.pattern = pattern
        self.max_queue = max_queue
        self.episode_len = episode_len
        self.switch_penalty = switch_penalty
        self.yellow_time = yellow_time
        self.discharge = discharge

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=max_queue, shape=(6,), dtype=np.float32
        )
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        return [seed]

    def _obs(self):
        # [qN, qS, qE, qW, phase, yellow_steps_left]
        return np.append(self.queues, [self.phase, self.yellow_left]).astype(np.float32)

    def _current_rates(self):
        if self.pattern == "rush":
            stage = min(2, self.t * 3 // self.episode_len)
            return np.array(self.RUSH_RATES[stage], dtype=np.float32)
        return self.arrival_rates

    def reset(self):
        self.queues = np.zeros(4, dtype=np.int32)
        self.phase = 0
        self.yellow_left = 0
        self.t = 0
        return self._obs()

    def step(self, action):
        switched = 0
        if self.yellow_left == 0 and action == 1:
            self.phase = 1 - self.phase
            self.yellow_left = self.yellow_time
            switched = 1

        if self.yellow_left > 0:
            self.yellow_left -= 1          # yellow: no cars leave
        else:
            green = [0, 1] if self.phase == 0 else [2, 3]
            for lane in green:
                self.queues[lane] = max(0, self.queues[lane] - self.discharge)

        arrivals = self.rng.poisson(self._current_rates())
        self.queues = np.minimum(self.queues + arrivals, self.max_queue)

        self.t += 1
        total_queue = int(self.queues.sum())
        reward = -total_queue - self.switch_penalty * switched
        done = self.t >= self.episode_len
        info = {"total_queue": total_queue, "switched": switched}
        return self._obs(), float(reward), done, info