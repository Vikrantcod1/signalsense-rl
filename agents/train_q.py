import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.intersection_env import IntersectionEnv
from agents.discretize import discretize
from agents.q_learning import QAgent
from baseline.fixed_timer import run_policy, fixed_timer

PATTERN = "rush"      # change to "rush" for the second run
EPISODES = 2000

os.makedirs("results", exist_ok=True)

env = IntersectionEnv(pattern=PATTERN)
env.seed(0)
agent = QAgent(seed=0)
returns = []

for ep in range(EPISODES):
    obs = env.reset()
    s = discretize(obs)
    done, ep_ret = False, 0.0
    while not done:
        a = agent.act(s)
        obs2, r, done, info = env.step(a)
        s2 = discretize(obs2)
        agent.update(s, a, r, s2)
        s, ep_ret = s2, ep_ret + r
    agent.decay()
    returns.append(ep_ret)
    if (ep + 1) % 200 == 0:
        print(f"episode {ep+1:4d} | avg return (last 200) {np.mean(returns[-200:]):8.1f} | eps {agent.eps:.2f}")

# learning curve
window = 50
smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
plt.figure(figsize=(7, 4))
plt.plot(smooth)
plt.xlabel("Episode")
plt.ylabel(f"Return (moving avg, {window})")
plt.title(f"Q-learning learning curve ({PATTERN})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"results/q_curve_{PATTERN}.png", dpi=150)
np.save(f"results/q_table_{PATTERN}.npy", agent.q)

# evaluation: greedy agent vs best fixed timer, same seed = same traffic
lam = env.arrival_rates.sum()
env.seed(42)
r, q = run_policy(env, lambda obs, t: agent.act(discretize(obs), greedy=True))
print(f"\nQ-learning | reward {r:8.1f} | avg queue {q:5.2f} | approx wait {q/lam:5.2f} steps")

best = None
for p in (6, 8, 10, 15, 20):
    env.seed(42)
    rf, qf = run_policy(env, fixed_timer(p))
    if best is None or rf > best[1]:
        best = (p, rf, qf)
print(f"Best fixed ({best[0]}) | reward {best[1]:8.1f} | avg queue {best[2]:5.2f} | approx wait {best[2]/lam:5.2f} steps")