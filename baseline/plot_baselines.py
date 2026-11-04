import os
import matplotlib.pyplot as plt
from env.intersection_env import IntersectionEnv
from baseline.fixed_timer import run_policy, fixed_timer

periods = [4, 5, 6, 8, 10, 15, 20, 30, 40, 60]
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 4))
for pattern in ("static", "rush"):
    env = IntersectionEnv(pattern=pattern)
    queues = []
    for p in periods:
        env.seed(42)
        _, q = run_policy(env, fixed_timer(p))
        queues.append(q)
    plt.plot(periods, queues, marker="o", label=pattern)

plt.xlabel("Fixed timer period (steps)")
plt.ylabel("Average total queue")
plt.title("Fixed-timer baseline: period sweep")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fixed_timer_sweep.png", dpi=150)
plt.show()