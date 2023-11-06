from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule, InteractionType
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, OneHotCategorical, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from torchrl.record.loggers.tensorboard import TensorboardLogger

from datetime import datetime
import os
import numpy as np

from CommonLearning import *
from Renderer import Renderer
from Model import Model

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, "valid")

torch.set_default_dtype(torch.float16)
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

# ---- Optimizer
lr = 1e-5
max_grad_norm = 1.0
weight_decay = 0.0
betas = (0.9, 0.999)

# ---- Saving/Loading
save_every = 10

save_path = "checkpointsPPO/" + str(datetime.now()) + "/"
os.makedirs(save_path)
save_path += "network_{}.ckpt"
CHECKPOINT_PATH = "checkpointsPPO/non_existent"


frame_skip = 5
# Adjust to fill VRAM
frames_per_batch = 2000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000_000 // frame_skip

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 3  # optimisation steps per batch of data collected
clip_epsilon = (
    0.1  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# ---- Replay
replay_size = frames_per_batch

Renderer.init()
env = create_environment(device, dtype=torch.float16, frame_skip=frame_skip)

print(env.observation_spec)

check_env_specs(env)


actor_net = Model(device, [
    nn.LazyLinear(env.action_spec.shape[-1], device=device),
]).to(device)

policy_module = TensorDictModule(
    actor_net, in_keys=["pixels", "linear_data"], out_keys=["logits"],
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
    default_interaction_type=InteractionType.RANDOM,
)

value_net = Model(device, [
    nn.LazyLinear(1, device=device),
]).to(device)

value_module = ValueOperator(
    module=value_net,
    in_keys=["pixels", "linear_data"],
)


print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

collector = SyncDataCollector(
    env,
    policy = policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    reset_at_each_iter=False,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(replay_size, device="cpu"),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
    normalize_advantage=True,
)

optim = torch.optim.Adam(
    loss_module.parameters(), 
    lr=lr,
    weight_decay=weight_decay,
    betas=betas,
    foreach=False
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)


pbar = tqdm(total=total_frames * frame_skip)
eval_str = ""

if not os.path.exists("checkpointsPPO/tensorboard/"):
    os.makedirs("checkpointsPPO/tensorboard/")
logger = TensorboardLogger(get_exp_name(
    type="ppo",
    date=datetime.now(),
), "checkpointsPPO/tensorboard/")

logger.log_hparams({
    "lr": lr,
    "max_grad_norm": max_grad_norm,
    "frame_skip": frame_skip,
    "frames_per_batch": frames_per_batch,
    "sub_batch_size": sub_batch_size,
    "num_epochs": num_epochs,
    "clip_epsilon": clip_epsilon,
    "gamma": gamma,
    "lmbda": lmbda,
    "entropy_eps": entropy_eps,
    "weight_decay": weight_decay,
    "beta0": betas[0],
    "beta1": betas[1],
    "replay_size": replay_size,
})

def save_state(epoch):
    global policy_module, actor_net, optim, logs, loss_module, advantage_module, collector, value_net, value_module, replay_buffer, scheduler
    path = save_path.format(epoch)
    torch.save(
        {
            "policy_module": policy_module.state_dict(),
            "actor_net": actor_net.state_dict(),
            "optim": optim.state_dict(),
            "loss_module": loss_module.state_dict(),
            "advantage_module": advantage_module.state_dict(),
            "collector": collector.state_dict(),
            "value_net": value_net.state_dict(),
            "value_module": value_module.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )
    print("Saved")
def load_state(path):
    global policy_module, actor_net, optim, logs, loss_module, advantage_module, collector, value_net, value_module, replay_buffer, scheduler
    loaded_state = torch.load(path)
    
    policy_module.load_state_dict(loaded_state["policy_module"])
    actor_net.load_state_dict(loaded_state["actor_net"])
    optim.load_state_dict(loaded_state["optim"])
    loss_module.load_state_dict(loaded_state["loss_module"])
    advantage_module.load_state_dict(loaded_state["advantage_module"])
    collector.load_state_dict(loaded_state["collector"])
    value_net.load_state_dict(loaded_state["value_net"])
    value_module.load_state_dict(loaded_state["value_module"])
    replay_buffer.load_state_dict(loaded_state["replay_buffer"])
    scheduler.load_state_dict(loaded_state["scheduler"])
    epoch = loaded_state["epoch"]

    print(f"Loaded epoch {epoch}")

if (os.path.exists(CHECKPOINT_PATH)):
    load_state(CHECKPOINT_PATH)


# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for epoch in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logger.log_scalar("reward", tensordict_data["next", "reward"].mean().item(), i)
    pbar.update(tensordict_data.numel() * frame_skip)
    logger.log_scalar("step count", tensordict_data["step_count"].max().item(), i)
    logger.log_scalar("lr", optim.param_groups[0]["lr"], i)
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our env horizon).
        # The ``rollout`` method of the env can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            env.reset()
            eval_rollout = env.rollout(5000, policy_module)
            logger.log_scalar("eval reward", eval_rollout["next", "reward"].mean().item(), i)
            logger.log_scalar("eval reward (sum)",
                eval_rollout["next", "reward"].sum().item(),
                i
            )
            logger.log_scalar("eval step count", eval_rollout["step_count"].max().item(), i)
            del eval_rollout
    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

    if (i % save_every == 0):
        save_state(i)

save_state("final")