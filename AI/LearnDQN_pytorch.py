import math
import random
from itertools import count
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

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
LR = 1e-3
WEIGHT_DECAY = 0.01

OPTIM_STEPS = 2
REPLAY_SIZE = 2000
FRAME_SKIP = 5

env = create_environment(device, torch.get_default_dtype(), "main", FRAME_SKIP, True)
# env = GymEnv("CartPole-v1", device=device)

# Get number of actions from gym action space
n_actions = env.action_spec.zero().flatten().shape[0]

policy_net = Model(n_actions).to(device)
# policy_net = nn.Sequential(
#     nn.LazyLinear(128),
#     nn.ReLU(),
#     nn.LazyLinear(128),
#     nn.ReLU(),
#     nn.LazyLinear(n_actions),
# )

policy_net = QValueActor(policy_net, in_keys=["pixels", "linear_data"], spec=env.action_spec)

state = env.reset()
policy_net(state)

loss_module = DQNLoss(policy_net, delay_value=True, action_space=env.action_spec)
target_updater = SoftUpdate(loss_module, tau=TAU)

optimizer = optim.AdamW(loss_module.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY, foreach=False, fused=True)

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
            return policy_net(state)
    else:
        state["action"] = env.action_spec.rand()
        return state

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = replay_buffer.sample(BATCH_SIZE).to(device)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)

    loss = loss_module(batch)
    loss["loss"].backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    target_updater.step()
    
save_dir = f"checkpointsDQN/{datetime.now():%m%d%Y %H:%M:%S}/"
save_filename = save_dir + "network_{}.ckpt"
CHECKPOINT_PATH = "checkpointsDQN/non_existent"

def save_state(epoch):
    global policy_net, target_net, optimizer, replay_buffer, criterion
    path = save_filename.format(epoch)
    torch.save(
        {
            "policy_net": policy_net.state_dict(),
            "loss_module": loss_module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "epoch": epoch
        },
        path
    )
    print("Saved")
def load_state(path):
    global policy_net, target_net, optimizer, replay_buffer, criterion
    loaded_state = torch.load(path)
    
    policy_net.load_state_dict(loaded_state["policy_net"])
    loss_module.load_state_dict(loaded_state["loss_module"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    replay_buffer.load_state_dict(loaded_state["replay_buffer"])
    epoch = loaded_state["epoch"]

    print(f"Loaded epoch {epoch}")

if (os.path.exists(CHECKPOINT_PATH)):
    load_state(CHECKPOINT_PATH)
os.makedirs(save_dir)


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


        logger.log_scalar("current game reward", reward_sum, t)

        if done:
            logger.log_scalar("reward", reward_sum, i_episode)
            logger.log_scalar("step count", t + 1, i_episode)
            break

    save_state(i_episode)

save_state("final")
print('Complete')