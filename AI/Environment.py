import numpy as np

import gymnasium as gym
from gymnasium import spaces

from Constants import *
from SocketController import SocketController
from Renderer import Renderer

import socket

class BlobEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "grayscale_array"]}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(1, ARENA_WIDTH, ARENA_HEIGHT), dtype=np.float32),
            spaces.Discrete(len(BLOB_RADII)),
            spaces.Discrete(len(BLOB_RADII)),
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        ))

        # We have 3 actions, corresponding to "right", "left", "drop"
        self.action_space = spaces.Discrete(3)

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

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.renderer = Renderer((ARENA_WIDTH, ARENA_HEIGHT), never_display=(render_mode != "human"))

        self.t = 0.5

        self.controller = None

    def _get_obs(self):
        self.renderer.render_frame(self.last_frame)
        current_blob = np.zeros(len(BLOB_RADII))
        current_blob[self.last_frame.current_blob] = 1
        next_blob = np.zeros(len(BLOB_RADII))
        next_blob[self.last_frame.next_blob] = 1
        
        return (
            np.array([np.array(self.renderer.get_pixels(), dtype=np.float32) / 255.0]),
            current_blob,
            next_blob,
            np.array([self.t])
        )

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if (not self.controller is None):
            self.controller.close_connection()
            self.controller.close_server()

        self.controller = SocketController(("localhost", 1234))
        self.controller.start_listening()
        self.controller.wait_for_connection()
        self.last_frame = self.controller.receive_frame_info()

        observation = self._get_obs()

        if self.render_mode == "human":
            self.renderer.display_frame()

        return observation, {}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.t = max(min(self.t + self._action_to_direction[action], 1), 0)
        
        should_drop = action == 2
        
        last_score = self.last_frame.score
        terminated = False
        
        try:
            self.controller.send_frame_info(self.t, should_drop)
            self.last_frame = self.controller.receive_frame_info()
            while not self.last_frame.can_drop:
                self.controller.send_frame_info(self.t, False)
                self.last_frame = self.controller.receive_frame_info()
        except socket.error:
            terminated = True

        # An episode is done if the agent has reached the target
        terminated = terminated or self.last_frame.is_game_over
        reward = self.last_frame.score - last_score if not terminated else -500

        if (reward < -100):
            terminated = True
        observation = self._get_obs()

        if self.render_mode == "human":
            self.renderer.display_frame()

        return observation, reward, terminated, False, {}

    def close(self):
        if self.controller is not None:
            self.controller.close_connection()
        
        if self.renderer is not None:
            self.renderer.close_window()