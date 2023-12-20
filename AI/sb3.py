import functools
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

from GymEnvironment import BlobEnvironment

if __name__ == "__main__":
    import random

    MOVE_STEP_SIZE = 0.01
    MOVE_PENALTY_THRESHOLD = (1.0 / MOVE_STEP_SIZE) * 2
    DROP_PENALTY_THRESHOLD = 5

    ALGORITHM = SAC

    ALG_NAME = ALGORITHM.__name__

    game_servers = [
        ("localhost", 32),
        # ("10.10.11.89", 32),
        # ("10.10.11.143", 24),
        # ("10.10.11.144", 24),
    ]

    # fallback_server = game_servers[0][0]

    game_servers = sum([[srv[0]]*srv[1] for srv in game_servers], [])
    random.shuffle(game_servers)

    print("Constructing environments...")
    env_fns = [
        functools.partial(
            BlobEnvironment,
            "train " + str(i),
            bool(i != 0),
            DROP_PENALTY_THRESHOLD,
            MOVE_PENALTY_THRESHOLD,
            MOVE_STEP_SIZE,
            is_eval=False,
            game_server = game_servers[i % len(game_servers)],
        )
        for i in range(64)
    ]
    env = VecMonitor(VecFrameStack(SubprocVecEnv(env_fns), n_stack=4))
    env_fns_eval = [
        functools.partial(
            BlobEnvironment,
            "eval " + str(i),
            bool(i != 0),
            DROP_PENALTY_THRESHOLD,
            MOVE_PENALTY_THRESHOLD,
            MOVE_STEP_SIZE,
            is_eval=True,
            game_server="localhost",
        )
        for i in range(8)
    ]
    eval_env = VecMonitor(VecFrameStack(SubprocVecEnv(env_fns_eval), n_stack=4))


    model = ALGORITHM(
        "MultiInputPolicy", env, verbose=1, tensorboard_log="./AI/tensorboard/", batch_size=256,
        learning_rate=5e-4
    )
    # model.set_parameters("./AI/dqn_success1_models/rl_model_dqn__9600000_steps.zip", exact_match=False)
    print("Started training")
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=f"{ALG_NAME}",
        progress_bar=True,
        callback=CallbackList(
            [
                CheckpointCallback(
                    100_000, f"./AI/models_{ALG_NAME}", verbose=2, name_prefix=f"rl_model_{ALG_NAME}_"
                ),
                EvalCallback(eval_env, eval_freq=5000, best_model_save_path=f"./AI/best_models_{ALG_NAME}/")
            ]
        ),
    )
    model.save(f"{ALG_NAME}_model.zip")

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")
