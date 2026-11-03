from env.intersection_env import IntersectionEnv

env = IntersectionEnv()
obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("Final obs:", obs)
print("Total reward:", total_reward)