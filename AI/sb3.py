import functools
from sbx import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from GymEnvironment import BlobEnvironment

MOVE_STEP_SIZE = 0.01
MOVE_PENALTY_THRESHOLD = (1.0 / MOVE_STEP_SIZE) * 1.5

if __name__ == "__main__":
    env_fns = [
        functools.partial(BlobEnvironment, str(i), True, MOVE_PENALTY_THRESHOLD, MOVE_STEP_SIZE)
        for i in range(16)
    ]
    env = VecMonitor(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))

    model = DQN(
        "MlpPolicy", env, verbose=1, tensorboard_log="./AI/a2c_cartpole_tensorboard/"
    )
    # model.load("model.zip")
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name="first_run",
        progress_bar=True,
        callback=CallbackList([CheckpointCallback(100_000, "./AI/sb3_models", verbose=2, name_prefix="rl_model_dqn_")]),
    )
    model.save("ppo_model.zip")

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")
