from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, OneHotDiscreteTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from Constants import *
from SocketController import SocketController
from Renderer import Renderer

import socket

from matplotlib import pyplot as plt

def first_nonone(arr, axis):
    mask = arr != 1

    result = np.full(arr.shape[axis], -1)

    indices = np.argmax(mask, axis=axis)

    result = np.take_along_axis(arr, np.expand_dims(indices, axis=axis), axis=axis)
    result = np.squeeze(result, axis=axis)

    return result

class BlobEnvironment(EnvBase):
    def __init__(self, dtype, worker_id, never_display, seed=None, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self.custom_dtype = dtype
        self.worker_id = worker_id
        self.observation_spec = CompositeSpec(
            pixels=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(NN_VIEW_WIDTH, NN_VIEW_HEIGHT),
                dtype=self.custom_dtype,
            ),
            # top_blob=BoundedTensorSpec(
            #     low=0,
            #     high=1,
            #     shape=(NN_VIEW_WIDTH),
            #     dtype=self.custom_dtype,
            # ),
            # top_distance=BoundedTensorSpec(
            #     low=0,
            #     high=1,
            #     shape=(NN_VIEW_WIDTH),
            #     dtype=self.custom_dtype,
            # ),
            can_drop=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(),
                dtype=self.custom_dtype,
            ),
            current_t=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(),
                dtype=self.custom_dtype,
            ),
            current_blob=OneHotDiscreteTensorSpec(
                n=len(BLOB_RADII),
                dtype=self.custom_dtype,
            ),
            next_blob=OneHotDiscreteTensorSpec(
                n=len(BLOB_RADII),
                dtype=self.custom_dtype,
            ),
            shape=(),
        )

        # We have 3 actions, corresponding to "drop", "right", "left"
        self.action_spec = OneHotDiscreteTensorSpec(
            n=3,
            dtype=self.custom_dtype,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))

        self._action_to_direction = {
            0: 0,
            1: -0.01,
            2: +0.01,
        }

        self.renderer = Renderer(never_display=never_display)

        self.t = 0.5
        self.controller = SocketController(("localhost", 1337), worker_id)

    def _get_obs(self, tensordict):
        self.renderer.render_frame(self.last_frame, self.t)

        pixels = self.renderer.get_pixels().astype(np.float16) / 255.0

        # rolled_pixels = np.roll(pixels, int(float(pixels.shape[-1]) * -self.t), -2)

        # top_blob = first_nonone(rolled_pixels, -1)
        # top_distance = (1.0-(np.argmax(rolled_pixels!=1, -1) / float(NN_VIEW_HEIGHT))) % 1.0
        
        # plt.imshow(np.moveaxis([np.concatenate([
        #     np.transpose(rolled_pixels),
        #     [[0]*100],
        #     [top_distance]*50,
        #     [[0]*100],
        #     [top_blob]*30,
        # ])]*3, [0, 1, 2], [2, 0, 1]))
        # plt.show(block=False)
        # plt.pause(0.01)

        
        return TensorDict(
            {
                "pixels": torch.tensor(pixels, dtype=self.custom_dtype, device=self.device),
                # "top_blob": torch.tensor(top_blob, dtype=self.custom_dtype, device=self.device),
                # "top_distance": torch.tensor(top_distance, dtype=self.custom_dtype, device=self.device),
                "current_blob": self.observation_spec["current_blob"].encode(self.last_frame.current_blob).to(self.device),
                "next_blob": self.observation_spec["next_blob"].encode(self.last_frame.next_blob).to(self.device),
                "current_t": torch.tensor(self.t, dtype=self.custom_dtype),
                "can_drop": torch.tensor(float(self.last_frame.can_drop), dtype=self.custom_dtype),
            },
            batch_size=(),
            device=self.device
        )

    def _reset(self, tensordict):
        self.controller.close_connection()

        self.controller = SocketController(("localhost", 1337), self.worker_id)
        self.last_frame = self.controller.receive_frame_info()

        observation = self._get_obs(tensordict)

        self.renderer.display_frame()

        return observation

    def _step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        
        self.t = (self.t + self._action_to_direction[torch.argmax(action["action"]).item()]) % 1.0
        
        should_drop = torch.argmax(action["action"]).item() == 0
        
        last_score = self.last_frame.score
        terminated = False
        
        try:
            self.controller.send_frame_info(self.t, should_drop)
            self.last_frame = self.controller.receive_frame_info()
        except socket.error:
            terminated = True

        # An episode is done if the agent has reached the target
        terminated = terminated or self.last_frame.is_game_over
        reward = self.last_frame.score - last_score if not terminated else -10

        if (reward < 0):
            terminated = True
        observation = self._get_obs(action)

        self.renderer.display_frame()

        observation["reward"] = float(reward)
        observation["done"] = terminated

        return observation

    def close(self):
        # self.controller.close_connection()
        # self.server.close_server()
        pass

    def _set_seed(self, seed: Optional[int]): pass
    
# env = BlobEnvironment()
# check_env_specs(env)