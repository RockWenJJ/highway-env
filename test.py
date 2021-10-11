import gym
import highway_env
import numpy as np

env = gym.make('intersection-hybrid-v0')
done = False
for ep in range(50):
    env.reset()
    while True:
        action = np.random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
