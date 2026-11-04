import numpy as np
from env.intersection_env import IntersectionEnv
from agents.discretize import discretize, N_LEVELS
from baseline.fixed_timer import fixed_timer

for pattern in ("static", "rush"):
    env = IntersectionEnv(pattern=pattern)
    env.seed(0)
    policy = fixed_timer(10)
    ns_counts = np.zeros(N_LEVELS, dtype=int)
    ew_counts = np.zeros(N_LEVELS, dtype=int)

    for _ in range(50):
        obs = env.reset()
        done, t = False, 0
        while not done:
            obs, _, done, _ = env.step(policy(obs, t))
            t += 1
            ns, ew, _, _ = discretize(obs)
            ns_counts[ns] += 1
            ew_counts[ew] += 1

    print(f"\nPattern: {pattern}")
    print("N-S level counts:", ns_counts)
    print("E-W level counts:", ew_counts)