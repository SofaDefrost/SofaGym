import os

from stable_baselines import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.sac import SAC
from stable_baselines.common import make_vec_env

for env_id in ["AntBulletEnv-v0"]:
    for seed in [1]:

        ########################### PPO ############################
        log_dir = "./logs/%s/AVEC-PPO_%s" % (env_id, seed)
        # log_dir = "./logs/%s/PPO_%s" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        model = PPO2('MlpPolicy', env, verbose=1, seed=seed,
                     avec_coef=1., vf_coef=0., tensorboard_log=log_dir)
        model.learn(total_timesteps=10000, tb_log_name="tb/AVEC-PPO")
        # model.learn(total_timesteps=1000000, tb_log_name="tb/PPO")


        ######################## TRPO ###########################
        log_dir = "./logs/%s/AVEC-TRPO_%s" % (env_id, seed)
        # log_dir = "./logs/%s/TRPO_%s" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        model = TRPO('MlpPolicy', env, verbose=1,
                     avec_coef=1., vf_coef=0., tensorboard_log=log_dir)
        model.learn(total_timesteps=10000, tb_log_name="tb/AVEC-TRPO")
        # model.learn(total_timesteps=1000000, tb_log_name="tb/TRPO")


        ######################### SAC #############################
        log_dir = "./logs/%s/AVEC-SAC_%s" % (env_id, seed)
        # log_dir = "./logs/%s/SAC_%s" % (env_id, seed)
        os.makedirs(log_dir, exist_ok=True)
        env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir)
        model = SAC('CustomSACPolicy', env, verbose=1,
                    avec_coef=1., value_coef=0., tensorboard_log=log_dir)
        model.learn(total_timesteps=10000, tb_log_name="tb/AVEC-SAC")
        # model.learn(total_timesteps=1000000, tb_log_name="tb/SAC")
