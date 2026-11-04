from env.intersection_env import IntersectionEnv
from baseline.fixed_timer import run_policy, fixed_timer

for pattern in ("static", "rush"):
    env = IntersectionEnv(pattern=pattern)
    lam = env.arrival_rates.sum()
    print(f"\nPattern: {pattern}")
    for period in (6, 8, 10, 15, 20):
        env.seed(42)
        r, q = run_policy(env, fixed_timer(period))
        print(f"Fixed ({period:2d}) | reward {r:8.1f} | avg queue {q:5.2f} | approx wait {q/lam:5.2f} steps")