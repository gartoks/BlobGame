import functools
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

from GymEnvironment import BlobEnvironment

MOVE_STEP_SIZE = 0.01
MOVE_PENALTY_THRESHOLD = (1.0 / MOVE_STEP_SIZE) * 2
DROP_PENALTY_THRESHOLD = 5

if __name__ == "__main__":
    env_fns = [
        functools.partial(
            BlobEnvironment, "train " + str(i), bool(i != 0), DROP_PENALTY_THRESHOLD, MOVE_PENALTY_THRESHOLD, MOVE_STEP_SIZE, is_eval=False
        )
        for i in range(16)
    ]
    env = VecMonitor(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))
    env_fns_eval = [
        functools.partial(
            BlobEnvironment, "eval " + str(i), bool(i != 0), DROP_PENALTY_THRESHOLD, MOVE_PENALTY_THRESHOLD, MOVE_STEP_SIZE, is_eval=True
        )
        for i in range(8)
    ]
    eval_env = VecMonitor(VecFrameStack(SubprocVecEnv(env_fns_eval), n_stack=4))


    model = PPO(
        "MultiInputPolicy", env, verbose=1, tensorboard_log="./AI/a2c_cartpole_tensorboard/", batch_size=128,
        learning_rate=5e-5
    )
    # model.set_parameters("./AI/dqn_success1_models/rl_model_dqn__9600000_steps.zip", exact_match=False)
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name="first_run",
        progress_bar=True,
        callback=CallbackList(
            [
                CheckpointCallback(
                    100_000, "./AI/sb3_models", verbose=2, name_prefix="rl_model_dqn_"
                ),
                EvalCallback(eval_env, eval_freq=5000, best_model_save_path="./AI/best_models/")
            ]
        ),
    )
    model.save("dqn_model.zip")

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")
