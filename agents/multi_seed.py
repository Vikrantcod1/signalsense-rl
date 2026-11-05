import numpy as np
from env.intersection_env import IntersectionEnv
from agents.discretize import discretize
from agents.q_learning import QAgent
from baseline.fixed_timer import run_policy, fixed_timer

SEEDS = [0, 1, 2, 3, 4]
EPISODES = 2000


def train(pattern, seed):
    env = IntersectionEnv(pattern=pattern)
    env.seed(seed)
    agent = QAgent(seed=seed)
    for _ in range(EPISODES):
        obs = env.reset()
        s = discretize(obs)
        done = False
        while not done:
            a = agent.act(s)
            obs2, r, done, info = env.step(a)
            s2 = discretize(obs2)
            agent.update(s, a, r, s2)
            s = s2
        agent.decay()
    return agent


for pattern in ("static", "rush"):
    env = IntersectionEnv(pattern=pattern)
    lam = env.arrival_rates.sum()

    # best fixed timer for this pattern
    best_r, best_q, best_p = None, None, None
    for p in (6, 8, 10, 15, 20):
        env.seed(42)
        r, q = run_policy(env, fixed_timer(p))
        if best_r is None or r > best_r:
            best_r, best_q, best_p = r, q, p
    fixed_wait = best_q / lam

    print(f"\n=== {pattern} | best fixed (period {best_p}) wait {fixed_wait:.2f} steps ===")
    waits, reductions = [], []
    for seed in SEEDS:
        agent = train(pattern, seed)
        env.seed(42)
        r, q = run_policy(env, lambda obs, t: agent.act(discretize(obs), greedy=True))
        wait = q / lam
        red = 100 * (1 - wait / fixed_wait)
        waits.append(wait)
        reductions.append(red)
        print(f"seed {seed} | reward {r:8.1f} | wait {wait:5.2f} steps | reduction {red:5.1f}%")

    print(f"SUMMARY {pattern}: wait {np.mean(waits):.2f} +/- {np.std(waits):.2f} steps, "
          f"reduction {np.mean(reductions):.1f}% +/- {np.std(reductions):.1f}%")