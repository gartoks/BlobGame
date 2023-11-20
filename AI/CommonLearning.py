
import torch

from Environment import BlobEnvironment

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

def create_environment(device, dtype, id, frame_skip=2, frame_stack=3, should_render=False):
    env = BlobEnvironment(dtype=dtype, device=device, worker_id=id, never_display=not should_render)

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
            FrameSkipTransform(frame_skip=frame_skip),
            # FlattenObservation(-2, -1, in_keys=["pixels"]),
            # UnsqueezeTransform(
            #     unsqueeze_dim=-1,
            #     in_keys=["can_drop", "current_t"],
            # ),
            # CatTensors(
            #     in_keys=["top_blob", "top_distance", "can_drop", "current_blob", "next_blob", "current_t"], dim=-1, out_key="linear_data",
            # ),
            UnsqueezeTransform(
                unsqueeze_dim=-3,
                in_keys=["pixels"],
            ),
            # CatFrames(N=frame_stack, dim=-1, in_keys=["linear_data"]),
            CatFrames(N=frame_stack, dim=-3, in_keys=["pixels"]),
        ),
    )

    return env

def get_exp_name(
        type,
        date,
        ):
    return f"{type=} {date}"

