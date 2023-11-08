
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

def create_environment(device, dtype, id, frame_skip=2, should_render=False):
    env = BlobEnvironment(dtype=dtype, device=device, worker_id=id, never_display=not should_render)

    env = TransformedEnv(
        env,
        Compose(
            StepCounter(),
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
                unsqueeze_dim=-3,
                in_keys=["pixels"],
            ),
            CatFrames(N=3, dim=-1, in_keys=["linear_data"]),
            CatFrames(N=3, dim=-3, in_keys=["pixels"]),
        ),
    )

    # env.transform[2].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return env

def get_exp_name(
        type,
        date,
        ):
    return f"{type=} {date}"

