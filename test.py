import gym
import highway_env
import numpy as np
import time

env = gym.make('intersection-hybrid-v0')
done = False
for ep in range(50):
    env.reset()
    while True:
        action = np.random.randint(0, 3)
        # action = np.random.random(2)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        if done:
            break
