import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from tensordict import TensorDict

from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.envs import GymEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GymEnv("CartPole-v1", render_mode="human").to(device)
env._env.env.env.metadata["render_fps"] = 999999
# if GPU is to be used

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        self.pipeline = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(n_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.pipeline(x["observation"])
    

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
WEIGHT_DECAY = 0.01

OPTIM_STEPS = 1
REPLAY_SIZE = 10000

# Get number of actions from gym action space
n_actions = env.action_spec.zero().flatten().shape[0]
# Get the number of state observations
state = env.reset()
n_observations = env.observation_spec.zero()["observation"].flatten().shape[0]

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
policy_net(state)
target_net(state)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY, foreach=False)

replay_buffer = TensorDictReplayBuffer(
    batch_size=BATCH_SIZE,
    storage=LazyTensorStorage(REPLAY_SIZE),
    prefetch=OPTIM_STEPS,
)


run_name = f"dqn {datetime.now():%m-%d%Y %H:%M:%S}"

if not os.path.exists("checkpointsDQN/tensorboard/"):
    os.makedirs("checkpointsDQN/tensorboard/")
logger = TensorboardLogger(run_name, "checkpointsDQN/tensorboard/")

logger.log_hparams({
    "lr": LR,
    "frame_skip": 1,
    "frames_per_batch": BATCH_SIZE,
    "sub_batch_size": BATCH_SIZE,
    "num_epochs": OPTIM_STEPS,
    "gamma": GAMMA,
    "tau": TAU,
    "weight_decay": WEIGHT_DECAY,
    "replay_size": REPLAY_SIZE,
})


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            state["action"] = env.action_spec.zero().scatter(0, torch.argmax(policy_net(state)), torch.ones((1), dtype=torch.int64, device=device))
    else:
        state["action"] = env.action_spec.rand()
    return state

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE).to(device)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    is_state_not_terminated = lambda s: not (s["done"] or s["terminated"])

    non_final_mask = torch.tensor(tuple(map(is_state_not_terminated, batch["next"])), device=device, dtype=torch.bool)
    non_final_next_states = batch["next"][non_final_mask]

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(batch).gather(1, torch.argmax(batch["action"], dim=1, keepdim=True))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + state["reward"]

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 600

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    reward_sum = 0
    for t in count():
        state = select_action(state)
        state = env.step(state)
        reward = state["next", "reward"]
        reward_sum += reward.item()
        done = state["next", "done"] or state["next", "terminated"]

        # Store the transition in memory
        replay_buffer.add(state)

        # Move to the next state
        state = state["next"]

        # Perform one step of the optimization (on the policy network)
        for i in range(OPTIM_STEPS):
            optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)



        if done:
            logger.log_scalar("reward", reward_sum, i_episode)
            logger.log_scalar("step count", t + 1, i_episode)
            break

print('Complete')