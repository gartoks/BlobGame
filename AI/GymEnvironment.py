from collections import defaultdict
from typing import Optional

import numpy as np
from Constants import *
from SocketController import SocketController
from Renderer import Renderer

import socket

from matplotlib import pyplot as plt

import gymnasium as gym
from gymnasium import spaces

def first_nonone(arr, axis):
    mask = arr != 1

    result = np.full(arr.shape[axis], -1)

    indices = np.argmax(mask, axis=axis)

    result = np.take_along_axis(arr, np.expand_dims(indices, axis=axis), axis=axis)
    result = np.squeeze(result, axis=axis)

    return result

class BlobEnvironment(gym.Env):
    def __init__(self, worker_id, never_display, move_penalty_threshold, move_step_size):
        super().__init__()
        self.worker_id = worker_id
        # self.observation_space = spaces.Box(low=0, high=255,
		# 									shape=(1, NN_VIEW_HEIGHT, NN_VIEW_WIDTH), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1,
											shape=(NN_VIEW_WIDTH + 1 + len(BLOB_RADII),), dtype=np.float32)


        # We have 3 actions, corresponding to "drop", "right", "left"
        self.action_space = spaces.Discrete(3)

        self._action_to_direction = {
            0: 0,
            1: -move_step_size,
            2: +move_step_size,
        }

        self.renderer = Renderer(never_display=never_display)

        self.t = 0.5
        self.controller = SocketController(("localhost", 1337), worker_id)

        self.num_moves_since_last_drop = 0
        self.move_penalty_threshold = move_penalty_threshold

    def _get_obs(self):
        self.renderer.render_frame(self.last_frame, self.t)

        pixels = self.renderer.get_pixels().astype(np.uint8)

        rolled_pixels = np.roll(pixels, int(float(pixels.shape[-1]) * -self.t), -2)

        top_blob = first_nonone(rolled_pixels, -1)
        top_distance = (1.0-(np.argmax(rolled_pixels!=1, -1) / float(NN_VIEW_HEIGHT))) % 1.0
        
        current_blob = np.zeros((len(BLOB_RADII)))
        current_blob[self.last_frame.current_blob] = 1
        # plt.imshow(np.moveaxis([np.concatenate([
        #     np.transpose(rolled_pixels),
        #     [[0]*100],
        #     [top_distance]*50,
        #     [[0]*100],
        #     [top_blob]*30,
        # ])]*3, [0, 1, 2], [2, 0, 1]))
        # plt.show(block=False)
        # plt.pause(0.01)

        pixels = np.expand_dims(np.transpose(pixels), axis=0)


        
        return np.concatenate([top_blob, current_blob, [self.t]])

    def reset(self, seed=0):
        self.controller.close_connection()

        self.controller = SocketController(("localhost", 1337), self.worker_id)
        self.last_frame = self.controller.receive_frame_info()
        self.num_moves_since_last_drop = 0

        observation = self._get_obs()

        self.renderer.display_frame()

        return observation, {}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        
        self.t = (self.t + self._action_to_direction[action]) % 1.0
        
        should_drop = action == 0
        if (should_drop):
            self.num_moves_since_last_drop = 0
        else:
            if (self.last_frame.can_drop):
                self.num_moves_since_last_drop += 1
        
        last_score = self.last_frame.score
        terminated = False
        
        try:
            self.controller.send_frame_info(self.t, should_drop)
            self.last_frame = self.controller.receive_frame_info()
        except socket.error:
            terminated = True

        # An episode is done if the agent has reached the target
        terminated = terminated or self.last_frame.is_game_over
        game_reward = self.last_frame.score - last_score
        reward = game_reward

        if (self.num_moves_since_last_drop >= self.move_penalty_threshold):
            reward -= 1

        observation = self._get_obs()

        self.renderer.display_frame()

        return observation, reward, terminated, False, {}

    def close(self):
        self.controller.close_connection()