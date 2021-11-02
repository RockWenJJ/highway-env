import gym
import highway_env
import sys
# from scripts.utils import show_videos
import wandb

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

env_config = "configs/IntersectionEnv/env_grid.json"
agent_config = "configs/IntersectionEnv/agents/DQNAgent/grid_convnet.json"

env = load_environment(env_config)
agent = load_agent(agent_config, env)

config = {'env': env.config, 'agent': agent.config}
wandblogger = wandb.init(project='test', config=config)
evaluation = Evaluation(env, agent, num_episodes=8000, display_env=False)

evaluation.train(wandblogger)