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

class BlobEnvironment(EnvBase):
    def __init__(self, seed=None, device="cpu"):
        super().__init__(device=device, batch_size=[])
        self.observation_spec = CompositeSpec(
            pixels=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(NN_VIEW_WIDTH, NN_VIEW_HEIGHT),
                dtype=torch.float64,
            ),
            can_drop=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(),
                dtype=torch.float64,
            ),
            current_t=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(),
                dtype=torch.float64,
            ),
            current_blob=OneHotDiscreteTensorSpec(
                n=len(BLOB_RADII),
                dtype=torch.float64,
            ),
            next_blob=OneHotDiscreteTensorSpec(
                n=len(BLOB_RADII),
                dtype=torch.float64,
            ),
            shape=(),
        )

        # We have 3 actions, corresponding to "right", "left", "drop"
        self.action_spec = OneHotDiscreteTensorSpec(
            n=3,
            dtype=torch.float64,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: -0.01,
            1: +0.01,
            2: 0
        }

        self.renderer = Renderer((ARENA_WIDTH, ARENA_HEIGHT), never_display=False)

        self.t = 0.5
        self.controller = None

    def _get_obs(self, tensordict):
        self.renderer.render_frame(self.last_frame)
        
        return TensorDict(
            {
                "pixels": torch.tensor(self.renderer.get_pixels(), dtype=torch.float64, device=self.device) / 255.0,
                "current_blob": self.observation_spec["current_blob"].encode(self.last_frame.current_blob).to(self.device),
                "next_blob": self.observation_spec["next_blob"].encode(self.last_frame.next_blob).to(self.device),
                "current_t": torch.tensor(self.t, dtype=torch.float64),
                "can_drop": torch.tensor(float(self.last_frame.can_drop), dtype=torch.float64),
            },
            batch_size=(),
            device=self.device
        )

    def _reset(self, tensordict):
        if (not self.controller is None):
            self.controller.close_connection()
            self.controller.close_server()

        self.controller = SocketController(("localhost", 1234))
        self.controller.start_listening()
        self.controller.wait_for_connection()
        self.last_frame = self.controller.receive_frame_info()

        observation = self._get_obs(tensordict)

        self.renderer.display_frame()

        return observation

    def _step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        
        
        self.t = (self.t + self._action_to_direction[torch.argmax(action["action"]).item()]) % 1.0
        
        should_drop = torch.argmax(action["action"]).item() == 2
        
        last_score = self.last_frame.score
        terminated = False
        
        try:
            self.controller.send_frame_info(self.t, should_drop)
            self.last_frame = self.controller.receive_frame_info()
        except socket.error:
            terminated = True

        # An episode is done if the agent has reached the target
        terminated = terminated or self.last_frame.is_game_over
        reward = self.last_frame.score - last_score if not terminated else -500

        if (reward < -100):
            terminated = True
        observation = self._get_obs(action)

        self.renderer.display_frame()

        observation["reward"] = float(reward)
        observation["done"] = terminated

        return observation

    def close(self):
        if self.controller is not None:
            self.controller.close_connection()
        
        if self.renderer is not None:
            self.renderer.close_window()

    def _set_seed(self, seed: Optional[int]): pass
    
# env = BlobEnvironment()
# check_env_specs(env)