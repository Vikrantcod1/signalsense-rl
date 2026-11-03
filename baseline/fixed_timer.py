import numpy as np
from env.intersection_env import IntersectionEnv


def run_policy(env, policy, episodes=100):
    rewards, avg_queues = [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward, queue_sum, t = 0.0, 0, 0
        while not done:
            action = policy(obs, t)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            queue_sum += info["total_queue"]
            t += 1
        rewards.append(ep_reward)
        avg_queues.append(queue_sum / t)
    return np.mean(rewards), np.mean(avg_queues)


def fixed_timer(period=30):
    # switch the signal every `period` steps
    return lambda obs, t: 1 if (t + 1) % period == 0 else 0


if __name__ == "__main__":
    env = IntersectionEnv()
    lam = env.arrival_rates.sum()  # total cars arriving per step

    env.seed(42)
    r, q = run_policy(env, lambda obs, t: env.action_space.sample())
    print(f"Random      | reward {r:8.1f} | avg queue {q:5.2f} | approx wait {q/lam:5.2f} steps")

    env.seed(42)
    r, q = run_policy(env, fixed_timer(30))
    print(f"Fixed (30)  | reward {r:8.1f} | avg queue {q:5.2f} | approx wait {q/lam:5.2f} steps")