import torch

from GymEnvironment import BlobEnvironment

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
    GymWrapper,
    ParallelEnv
)

import functools

def create_environment(device, dtype, id, frame_skip=2, frame_stack=3, move_penalty_threshold=100, move_step_size = 0.001, drop_penalty_threshold=5, is_eval=False):
    def env_fn(i):
        env = BlobEnvironment(worker_id=str(id) + " " + str(i), never_display=(i!=0), move_penalty_threshold=move_penalty_threshold, move_step_size=move_step_size, drop_penalty_threshold=drop_penalty_threshold, is_eval=is_eval)

        env = GymWrapper(env)
        return env
    
    env = ParallelEnv(16, [functools.partial(env_fn, i) for i in range(16)])

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            FrameSkipTransform(frame_skip=frame_skip),
            DTypeCastTransform(dtype_in=torch.uint8, dtype_out=torch.float32, in_keys=["pixels"]),
            # FlattenObservation(-2, -1, in_keys=["pixels"]),
            # UnsqueezeTransform(
            #     unsqueeze_dim=-1,
            #     in_keys=["can_drop", "current_t"],
            # ),
            # CatTensors(
            #     in_keys=["top_blob", "top_distance", "can_drop", "current_blob", "next_blob", "current_t"], dim=-1, out_key="linear_data",
            # ),
            # UnsqueezeTransform(
            #     unsqueeze_dim=-3,
            #     in_keys=["pixels"],
            # ),
            # CatFrames(N=frame_stack, dim=-1, in_keys=["linear_data"]),
            CatFrames(N=frame_stack, dim=-3, in_keys=["pixels"]),
        ),
    )

    env.to(device)

    return env

