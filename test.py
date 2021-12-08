import gym
import highway_env
import numpy as np
import time
import yaml
import importlib
import matplotlib.pyplot as plt
from collections import deque


def load_environments(config):
    env_config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

    env = gym.make(env_config['id'])
    env.import_module = env_config.get('import_module', None)
    env.unwrapped.configure(env_config)
    env.reset()
    return env


if __name__ == '__main__':
    env = load_environments("configs/IntersectionEnv/env_nostop.json")
    done = False

    success_rate = deque(maxlen=101)
    vehicle_num = deque(maxlen=101)
    ave_steps = deque(maxlen=101)
    for ep in range(101):
        ep_reward = 0
        obs = env.reset()
        indexes = []
        obs_x = [obs[0][1]]
        obs_y = [obs[0][2]]
        v_nums = []
        step = 0
        while True:
            action = np.random.randint(0, 3)
            # action = np.random.random(2)
            next_obs, reward, done, info = env.step(action)
            v_nums.append(len(env.road.vehicles))
            # env.render()
            obs = next_obs
            obs_x.append(obs[0][1])
            obs_y.append(obs[0][2])
            ep_reward += reward
            # time.sleep(0.01)
            step += 1
            if done:
                if "agents_crashed" in info.keys():
                    crashed = info["agents_crashed"][0]
                    success = info["agent_arrived"] and not crashed
                else:
                    success = info.get("agent_arrived", 0)
                success_rate.append(success)
                print("Episode {}, success_rate {}， vehicle_num {}, steps {}"
                      .format(ep+1,  np.mean(success_rate), np.mean(v_nums), step))
                vehicle_num.append(np.mean(v_nums))
                if success:
                    ave_steps.append(step)
                # print("Episodic Reward: %.2f"%ep_reward)
                break

    print("Average vehicle num in the environment {}.".format(np.mean(vehicle_num)))
    print("Average Steps used {}.".format(np.mean(ave_steps)))

        # plt.plot(obs_x, obs_y)
        # plt.show()

