import gym
import highway_env
import numpy as np
import time
import yaml
import importlib
import matplotlib.pyplot as plt


def load_environments(config):
    env_config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

    env = gym.make(env_config['id'])
    env.import_module = env_config.get('import_module', None)
    env.unwrapped.configure(env_config)
    env.reset()
    return env


if __name__ == '__main__':
    env = load_environments("configs/IntersectionEnv/env_no_normalize.json")
    done = False
    for ep in range(50):
        ep_reward = 0
        obs = env.reset()
        indexes = []
        obs_x = [obs[0][1]]
        obs_y = [obs[0][2]]
        while True:
            action = np.random.randint(0, 3)
            # action = np.random.random(2)
            next_obs, reward, done, info = env.step(action)
            env.render()
            obs = next_obs
            obs_x.append(obs[0][1])
            obs_y.append(obs[0][2])
            ep_reward += reward
            time.sleep(0.01)
            if done:
                print("Episodic Reward: %.2f"%ep_reward)
                break

        # plt.plot(obs_x, obs_y)
        # plt.show()

