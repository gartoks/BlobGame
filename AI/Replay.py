import random, datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack

from Metrics import MetricLogger
from Agent import Agent
from Wrappers import SkipFrame

from Constants import *

from gymnasium.envs.registration import register

register(
     id="BlobGame-v0",
     entry_point="Environment:BlobEnvironment",
     max_episode_steps=None,
     nondeterministic=True,
)
env = gym.make('BlobGame-v0', render_mode="human")

env = SkipFrame(env, skip=4)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/agent.ckpt')
agent = Agent(image_dim=(1, ARENA_WIDTH, ARENA_HEIGHT), additional_dim=(len(BLOB_RADII)*2 + 1), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=None)
agent.exploration_rate = agent.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state, _ = env.reset()

    while True:

        action = agent.act(state)
        next_state, reward, terminated, done, info = env.step(action)
        agent.cache(state, next_state, action, reward, done)
        logger.log_step(reward, None, None)

        state = next_state

        if done or terminated:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )