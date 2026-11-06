from env.intersection_env import IntersectionEnv
from agents.discretize import discretize
from agents.q_learning import QAgent
from baseline.fixed_timer import fixed_timer

EPISODES = 4000


def train(pattern, seed=1):
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


def inspect(env, policy, episodes=100):
    ns_green = ew_green = yellow = switches = 0
    for _ in range(episodes):
        obs = env.reset()
        done, t = False, 0
        while not done:
            yl = env.yellow_left                 # yellow status before the step
            obs, r, done, info = env.step(policy(obs, t))
            t += 1
            switches += info["switched"]
            if yl > 0 or info["switched"]:
                yellow += 1
            elif env.phase == 0:
                ns_green += 1
            else:
                ew_green += 1
    total = ns_green + ew_green + yellow
    return (100 * ns_green / total, 100 * ew_green / total,
            100 * yellow / total, switches / episodes)


pattern = "static"
env = IntersectionEnv(pattern=pattern)
agent = train(pattern)

print(f"Pattern: {pattern}  (N-S carries 3x the traffic of E-W)\n")
print("policy         | N-S green | E-W green | yellow | switches/episode")

env.seed(42)
a = inspect(env, lambda obs, t: agent.act(discretize(obs), greedy=True))
print(f"Q-learning     | {a[0]:8.1f}% | {a[1]:8.1f}% | {a[2]:5.1f}% | {a[3]:6.1f}")

env.seed(42)
f = inspect(env, fixed_timer(10))
print(f"Fixed (10)     | {f[0]:8.1f}% | {f[1]:8.1f}% | {f[2]:5.1f}% | {f[3]:6.1f}")