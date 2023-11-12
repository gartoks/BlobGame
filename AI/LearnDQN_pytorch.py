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
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.envs import GymEnv

from CommonLearning import *
from Model import Model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

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

OPTIM_STEPS = 2
REPLAY_SIZE = 2000
FRAME_SKIP = 5

env = create_environment(device, torch.get_default_dtype(), "main", FRAME_SKIP, True)

# Get number of actions from gym action space
n_actions = env.action_spec.zero().flatten().shape[0]
# Get the number of state observations

def unfold(state):
    return state["pixels"], state["linear_data"]

policy_net = Model(n_actions).to(device)
target_net = Model(n_actions).to(device)

state = env.reset()
policy_net(*unfold(state))
target_net(*unfold(state))

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY, foreach=False, fused=True)

replay_buffer = TensorDictReplayBuffer(
    batch_size=BATCH_SIZE,
    storage=LazyTensorStorage(REPLAY_SIZE, device=device),
    prefetch=OPTIM_STEPS,
)


run_name = f"dqn {datetime.now():%m-%d%Y %H:%M:%S}"

if not os.path.exists("checkpointsDQN/tensorboard/"):
    os.makedirs("checkpointsDQN/tensorboard/")
logger = TensorboardLogger(run_name, "checkpointsDQN/tensorboard/")

logger.log_hparams({
    "lr": LR,
    "frame_skip": FRAME_SKIP,
    "frames_per_batch": BATCH_SIZE,
    "sub_batch_size": BATCH_SIZE,
    "num_epochs": OPTIM_STEPS,
    "gamma": GAMMA,
    "tau": TAU,
    "weight_decay": WEIGHT_DECAY,
    "replay_size": REPLAY_SIZE,
})

criterion = nn.SmoothL1Loss()


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
            state["action"] = policy_net(*unfold(state))
    else:
        state["action"] = env.action_spec.rand()
    return state

@torch.jit.script
def fused_optimize_model_(*, batch_next_done, batch_action, batch_pixels, batch_linear_data, state_reward, next_states_pixels, next_states_linear_data, BATCH_SIZE: int, GAMMA: float):
    non_final_mask = ~batch_next_done.squeeze()
    non_final_next_states_pixels = next_states_pixels[non_final_mask]
    non_final_next_states_linear_data = next_states_linear_data[non_final_mask]

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(batch_pixels, batch_linear_data).gather(1, torch.argmax(batch_action, dim=1, keepdim=True))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device="cuda")
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states_pixels, non_final_next_states_linear_data).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + state_reward

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(-1))

    return loss

def fused_optimize_model(**kwargs):
    return fused_optimize_model_(
        BATCH_SIZE=BATCH_SIZE,
        GAMMA=GAMMA,
        **kwargs
    )

is_first = True
def optimize_model():
    global batch, is_first
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE).to(device)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    
    loss = fused_optimize_model(
        batch_next_done=batch["next", "done"],
        batch_action=batch["action"],
        batch_linear_data=batch["linear_data"],
        batch_pixels=batch["pixels"],
        state_reward=state["reward"],
        next_states_linear_data=batch["next", "linear_data"],
        next_states_pixels=batch["next", "pixels"],
    )

    # Optimize the model
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
save_dir = f"checkpointsDQN/{datetime.now():%m%d%Y %H:%M:%S}/"
save_filename = save_dir + "network_{}.ckpt"
CHECKPOINT_PATH = "checkpointsDQN/non_existent"

def save_state(epoch):
    global policy_net, target_net, optimizer, replay_buffer, criterion
    path = save_filename.format(epoch)
    torch.save(
        {
            "policy_net": policy_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "criterion": criterion.state_dict(),
            "epoch": epoch
        },
        path
    )
    print("Saved")
def load_state(path):
    global policy_net, target_net, optimizer, replay_buffer, criterion
    loaded_state = torch.load(path)
    
    policy_net.load_state_dict(loaded_state["policy_net"])
    target_net.load_state_dict(loaded_state["target_net"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    replay_buffer.load_state_dict(loaded_state["replay_buffer"])
    criterion.load_state_dict(loaded_state["criterion"])
    epoch = loaded_state["epoch"]

    print(f"Loaded epoch {epoch}")

os.makedirs(save_filename)

if (os.path.exists(CHECKPOINT_PATH)):
    load_state(CHECKPOINT_PATH)


num_episodes = 2000

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    reward_sum = 0
    for t in count():
        state = select_action(state)
        state = env.step(state)
        reward = state["next", "reward"]
        reward_sum += reward.item()
        done = state["next", "done"]

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


        logger.log_scalar("current game reward", reward_sum, t)

        if done:
            logger.log_scalar("reward", reward_sum, i_episode)
            logger.log_scalar("step count", t + 1, i_episode)
            break

    save_state(i_episode)
print('Complete')