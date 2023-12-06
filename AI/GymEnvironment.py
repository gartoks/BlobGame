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
    def __init__(
        self,
        worker_id,
        never_display,
        drop_penalty_threshold,
        move_penalty_threshold,
        move_step_size,
        is_eval=False,
    ):
        super().__init__()
        self.worker_id = worker_id
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, NN_VIEW_HEIGHT, NN_VIEW_WIDTH),
                    dtype=np.uint8,
                ),
                "top_blob": spaces.Box(
                    low=0,
                    high=1,
                    shape=(NN_VIEW_WIDTH,),
                    dtype=np.float32,
                ),
                "top_distance": spaces.Box(
                    low=0,
                    high=1,
                    shape=(NN_VIEW_WIDTH,),
                    dtype=np.float32,
                ),
                "current_blob": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(BLOB_RADII),),
                    dtype=np.float32,
                ),
                "next_blob": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(BLOB_RADII),),
                    dtype=np.float32,
                ),
                "can_drop": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "t": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "drop_time_before_punishment": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "move_time_before_punishment": spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        # We have 3 actions, corresponding to "drop", "right", "left"
        self.action_space = spaces.Discrete(3)

        self._action_to_direction = {
            0: 0,
            1: -move_step_size,
            2: +move_step_size,
        }

        self.renderer = Renderer(
            never_display=never_display, window_title=self.worker_id
        )

        self.t = 0.5
        self.controller = SocketController(("localhost", 1337), worker_id)

        self.num_moves_since_last_drop = 0
        self.num_drops_since_last_move = 0
        self.move_penalty_threshold = move_penalty_threshold
        self.drop_penalty_threshold = drop_penalty_threshold

        self.is_eval = is_eval
        self.frame_count = 0

    def _get_obs(self):
        self.renderer.render_frame(self.last_frame, self.t)

        pixels = self.renderer.get_pixels().astype(np.float32) / 255.0

        rolled_pixels = np.roll(
            pixels, int(float(pixels.shape[-2]) * -(self.t + 0.5)), -2
        )

        rolled_cropped_pixels = rolled_pixels[:, 2 * NN_NEXT_BLOB_HEIGHT :]

        top_blob = first_nonone(rolled_cropped_pixels, -1)
        top_distance = (
            1.0 - (np.argmax(rolled_cropped_pixels != 1, -1) / float(NN_VIEW_HEIGHT))
        ) % 1.0

        current_blob = np.zeros((len(BLOB_RADII)))
        current_blob[self.last_frame.current_blob] = 1
        next_blob = np.zeros((len(BLOB_RADII)))
        next_blob[self.last_frame.next_blob] = 1

        # if (not self.renderer.never_display):
        #     plt.imshow(np.moveaxis([np.concatenate([
        #         np.transpose(rolled_pixels),
        #         [[0]*NN_VIEW_WIDTH],
        #         [top_distance]*50,
        #         [[0]*NN_VIEW_WIDTH],
        #         [top_blob]*30,
        #     ])]*3, [0, 1, 2], [2, 0, 1]))
        #     plt.show(block=False)
        #     plt.pause(0.01)

        pixels = np.expand_dims(np.transpose(pixels), axis=0)

        return {
            "pixels": (pixels * 255).astype(np.uint8),
            "top_blob": top_blob.astype(np.float32),
            "top_distance": top_distance.astype(np.float32),
            "current_blob": current_blob.astype(np.float32),
            "next_blob": next_blob.astype(np.float32),
            "t": np.array([self.t], dtype=np.float32),
            "can_drop": np.array([float(self.last_frame.can_drop)], dtype=np.float32),
            "drop_time_before_punishment": np.array(
                [self.num_drops_since_last_move / self.drop_penalty_threshold],
                dtype=np.float32,
            ),
            "move_time_before_punishment": np.array(
                [self.num_moves_since_last_drop / self.move_penalty_threshold],
                dtype=np.float32,
            ),
        }

    def reset(self, seed=0):
        self.controller.close_connection()

        self.controller = SocketController(("localhost", 1337), self.worker_id)
        self.last_frame = self.controller.receive_frame_info()
        self.num_moves_since_last_drop = 0
        self.num_drops_since_last_move = 0

        observation = self._get_obs()

        self.renderer.display_frame()
        self.frame_count = 0
        self.last_top_blob = [0] * NN_VIEW_WIDTH

        return observation, {}

    def step(self, action):
        ## Bookkeeping
        self.t = (self.t + self._action_to_direction[action]) % 1.0

        should_drop = action == 0
        if should_drop:
            self.num_moves_since_last_drop = 0
            self.num_drops_since_last_move += 1
        else:
            if self.last_frame.can_drop:
                self.num_moves_since_last_drop += 1
                self.num_drops_since_last_move = 0

        terminated = False
        self.frame_count += 1

        ## Updating
        try:
            self.controller.send_frame_info(self.t, should_drop)
            new_frame = self.controller.receive_frame_info()
        except socket.error:
            terminated = True
        terminated |= new_frame.is_game_over

        ## Observe
        observation = self._get_obs()
        self.renderer.display_frame()

        ## Failsafe
        terminated |= self.frame_count > 20_000
        terminated |= self.num_moves_since_last_drop > 300
        terminated |= self.num_drops_since_last_move > 300

        ## Reward
        game_reward = new_frame.score - self.last_frame.score
        reward = game_reward

        position_badness = 0
        for blob in self.last_frame.blobs:
            if blob.type >= 3:
                position_badness += (blob.y / ARENA_HEIGHT) * BLOB_RADII[blob.type - 3]

        if len(self.last_frame.blobs):
            position_badness /= len(self.last_frame.blobs)

        reward -= position_badness / 500

        if self.num_moves_since_last_drop >= self.move_penalty_threshold:
            reward -= 10
        if self.num_drops_since_last_move >= self.move_penalty_threshold:
            reward -= 10

        if should_drop and self.last_frame.can_drop:
            cblob_color = BLOB_COLORS[self.last_frame.current_blob][0] / 255
            if self.last_top_blob[NN_VIEW_WIDTH // 2] == cblob_color:
                reward += 20
            elif np.isin(cblob_color, self.last_top_blob):
                reward -= 5

        if terminated:
            reward -= 100

        reward = reward / 20

        self.last_top_blob = observation["top_blob"]
        self.last_frame = new_frame

        if self.is_eval:
            return (
                observation,
                game_reward,
                terminated,
                False,
                {},
            )
        else:
            return (
                observation,
                reward,
                terminated,
                False,
                {},
            )

    def close(self):
        self.controller.close_connection()
    

if __name__ == "__main__":
    import time
    from stable_baselines3.common.env_checker import check_env
    from tqdm.rich import tqdm

    env = BlobEnvironment("0", False, 100, 5, 0.01, False)
    env.reset()

    check_env(env)

    N_IT = 10_000
    st = time.perf_counter()
    for i in tqdm(range(N_IT)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # time.sleep(1 / 120)
        # print(reward)
        if terminated or truncated:
            env.reset()

    et = time.perf_counter()
    print(f"{N_IT} iterations took {et-st:.2f}s ({N_IT/(et-st):.3f}it/s)")
