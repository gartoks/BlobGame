import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

# env = SkipFrame(env, skip=4)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None #Path('checkpoints/2023-10-30T00-37-29/mario_net_10.chkpt')
agent = Agent(image_dim=(1, ARENA_WIDTH, ARENA_HEIGHT), additional_dim=(len(BLOB_RADII)*2 + 1), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=None)

logger = MetricLogger(save_dir)

episodes = 40000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state, _ = env.reset()
    agent.reset_feedback()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = agent.act(state)

        # 5. Agent performs action
        next_state, reward, terminated, done, info = env.step(action)

        # 6. Remember
        agent.cache(state, next_state, action, reward, done or terminated)

        # 7. Learn
        q, loss = agent.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or terminated:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )