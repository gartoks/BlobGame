from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    FrameSkipTransform,
    UnsqueezeTransform,
    CatTensors,
    FlattenObservation,
    CatFrames,
    DTypeCastTransform,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, OneHotCategorical, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from datetime import datetime
import os
import numpy as np

from Environment import BlobEnvironment
from Renderer import Renderer
from Model import Model

def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, "valid")

torch.set_default_dtype(torch.float32)

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
lr = 3e-4
max_grad_norm = 1.0
save_every = 100

save_path = "checkpoints/" + str(datetime.now()) + "/"
os.makedirs(save_path)
save_path += "network_{}.ckpt"
CHECKPOINT_PATH = "checkpoints/non_existent"

frame_skip = 2
# Adjust to fill VRAM
frames_per_batch = 2000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 1_000_000_000 // frame_skip

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

def create_environment():
    env = BlobEnvironment(device=device)

    env = TransformedEnv(
        env,
        Compose(
            FrameSkipTransform(frame_skip=frame_skip),
            # FlattenObservation(-2, -1, in_keys=["pixels"]),
            UnsqueezeTransform(
                unsqueeze_dim=-1,
                in_keys=["can_drop", "current_t"],
            ),
            CatTensors(
                in_keys=["top_blob", "top_distance", "can_drop", "current_blob", "next_blob", "current_t"], dim=-1, out_key="linear_data",
            ),
            # normalize observations

            # ObservationNorm(in_keys=["observation"]),
            # DoubleToFloat(
            #     in_keys=["observation"],
            # ),
            # UnsqueezeTransform(
            #     unsqueeze_dim=-2,
            #     in_keys=["observation"],
            #     in_keys_inv=["observation"],
            # ),
            UnsqueezeTransform(
                unsqueeze_dim=-1,
                in_keys=["linear_data"],
            ),
            UnsqueezeTransform(
                unsqueeze_dim=-3,
                in_keys=["pixels"],
            ),
            CatFrames(N=3, dim=-1, in_keys=["linear_data"]),
            CatFrames(N=3, dim=-3, in_keys=["pixels"]),
            StepCounter(),
        ),
    )

    # env.transform[2].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return env

Renderer.init()
env = create_environment()

print(env.observation_spec)

check_env_specs(env)


actor_net = Model([
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
)

value_net = Model([
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
    storage=LazyTensorStorage(frames_per_batch, device="cpu"),
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
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)


logs = defaultdict(list)
pbar = tqdm(total=total_frames * frame_skip)
eval_str = ""

def save_state(epoch):
    global policy_module, actor_net, optim, logs, loss_module, advantage_module, collector, value_net, value_module, replay_buffer, scheduler
    path = save_path.format(epoch)
    torch.save(
        {
            "policy_module": policy_module.state_dict(),
            "actor_net": actor_net.state_dict(),
            "optim": optim.state_dict(),
            "logs": logs,
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
    
    policy_module.load_state_dict(load_state["policy_module"])
    actor_net.load_state_dict(loaded_state["actor_net"])
    optim.load_state_dict(loaded_state["optim"])
    logs = loaded_state["logs"]
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

matplotlib.use("Qt5agg")
plt.figure(figsize=(15, 10))
plt.show(block=False)

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

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel() * frame_skip)
    cum_reward_str = (
        f"⌀ reward={logs['reward'][-1]: 4.4f} ({logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"steps (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our env horizon).
        # The ``rollout`` method of the env can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"∑ reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"({logs['eval reward (sum)'][0]: 4.4f}), "
                f"steps: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

    plt.clf()
    plt.subplot(3, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(3, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(3, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(3, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.subplot(3, 2, 5)
    plt.plot(moving_average(logs["reward"], 100))
    plt.title("Reward (moving average)")
    plt.pause(0.01)

    if (i % save_every == 0):
        save_state(i)

plt.show()
save_state("final")