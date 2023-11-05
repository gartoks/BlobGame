import os

import torch
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer

from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.trainers import (
    
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
from torchrl.envs.utils import ExplorationType

from CommonLearning import *
from Model import Model
from Renderer import Renderer

from datetime import datetime

Renderer.init()

#-----------------
# torch.set_default_dtype(torch.float16)
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# the learning rate of the optimizer
lr = 2e-4
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8

gamma = 0.99
tau = 0.02

total_frames = 100_000
frame_skip = 5
init_random_frames = 1000
frames_per_batch = 256
batch_size = 256
buffer_size = min(total_frames, 100000)

eps_greedy_val = 0.2
eps_greedy_val_env = 0.005

init_bias = 2.0
#-----------------


def make_model(dummy_env):
    net = Model(device, [
        nn.LazyLinear(dummy_env.action_spec.shape[-1], device=device),
    ]).to(device)
    net.combined_pipeline[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels", "linear_data"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict().to(device)
    actor(tensordict)

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return actor, actor_explore

def get_replay_buffer(buffer_size, n_optim, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyMemmapStorage(buffer_size),
        prefetch=n_optim,
    )
    return replay_buffer


def get_collector(
    actor_explore,
    frames_per_batch,
    total_frames,
    env,
    device,
):
    data_collector = SyncDataCollector(
        env,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector

def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater

test_env = create_environment(device, dtype=torch.float32, frame_skip=frame_skip)
# Get model
actor, actor_explore = make_model(test_env)
loss_module, target_net_updater = get_loss_module(actor, gamma)

collector = get_collector(
    actor_explore=actor_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    env=test_env,
    device=device,
)
optimizer = torch.optim.Adam(
    loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
)

save_path = "checkpointsDQN/" + str(datetime.now()) + "/"
os.makedirs(save_path)

logger = TensorboardLogger(exp_name="dqn_exp", log_dir=save_path)
log_interval = 500

trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=1,
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,
    optim_steps_per_batch=n_optim,
    log_interval=log_interval,
)

buffer_hook = ReplayBufferTrainer(
    get_replay_buffer(buffer_size, n_optim, batch_size=batch_size),
    flatten_tensordicts=True,
)
buffer_hook.register(trainer)
weight_updater = UpdateWeights(collector, update_weights_interval=1)
weight_updater.register(trainer)
recorder = Recorder(
    record_interval=100,  # log every 100 optimization steps
    record_frames=1000,  # maximum number of frames in the record
    frame_skip=frame_skip,
    policy_exploration=actor_explore,
    environment=test_env,
    exploration_type=ExplorationType.MODE,
    log_keys=[("next", "reward")],
    out_keys={("next", "reward"): "rewards"},
    log_pbar=True,
)
recorder.register(trainer)

trainer.register_op("post_optim", target_net_updater.step)

log_reward = LogReward(log_pbar=True)
log_reward.register(trainer)


trainer.train()