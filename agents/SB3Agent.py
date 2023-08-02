import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

import torch as th
import yaml
from agents.SofaBaseAgent import SofaBaseAgent
from agents.utils import make_env, mkdirp, sec_to_hours
from colorama import Fore
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecVideoRecorder,
                                              sync_envs_normalization)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


# Adapted from RL Baselines3 Zoo
def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
        if key not in hyperparams or hyperparams[key] is None:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class SB3Agent(SofaBaseAgent):
    def __init__(self, env_id, algo_name, seed=0, output_dir="./Results", max_episode_steps=None, n_envs=1, model_name=None, **kwargs):
        super().__init__(env_id, seed, output_dir, max_episode_steps, n_envs)
        
        self.algo_name = algo_name
        self.algo = eval(self.algo_name)
        self.model_name = model_name
        self.params = kwargs

        self.init_model()

    def init_dirs(self):
        if self.model_name is None:
            self.model_time = int(time.time())
            self.model_name = str(self.seed) + "_" + str(self.model_time)
        
        self.model_dir = f"{self.output_dir}/{self.env_id}/{self.algo_name}/{self.model_name}"
        self.checkpoints_dir = f"{self.output_dir}/{self.env_id}/{self.algo_name}/{self.model_name}/model"
        mkdirp(self.checkpoints_dir)

        self.log_dir = f"{self.model_dir}/log"
        mkdirp(self.log_dir)
        
        self.video_dir = f"{self.log_dir}/videos"
        mkdirp(self.video_dir)

        self.stats_path = f"{self.log_dir}/vec_normalize.pkl"
        self.model_log_path = f"{self.log_dir}/model_log.pkl"

    def load_params(self):
        if not self.params:
            self.params_path = "./agents/hyperparameters/stable_baselines_params.yml"
            config = yaml.safe_load(Path(self.params_path).read_text())
            self.params = config[self.algo_name]
        
        self.init_kwargs = self.params['init_kwargs'].copy()
        if self.init_kwargs.get('policy_kwargs'):
            self.init_kwargs['policy_kwargs'] = eval(self.init_kwargs['policy_kwargs'])
        if self.init_kwargs.get('train_freq'):
            self.init_kwargs['train_freq'] = eval(self.init_kwargs['train_freq'])
        
        self.init_kwargs = _preprocess_schedules(self.init_kwargs)

        self.fit_kwargs = self.params['fit_kwargs']

        if self.max_episode_steps is None:
            self.max_episode_steps = self.fit_kwargs['max_episode_steps']
    
    def init_model(self):
        self.load_params()
        self.init_dirs()

        self.env = self.env_wrap(self.n_envs, normalize=True)

        checkpoint_path = f"{self.checkpoints_dir}/latest"
        if not os.path.exists(checkpoint_path + ".zip"):
            self.model = self.algo(env=self.env, seed=self.seed, verbose=1, tensorboard_log=self.log_dir, **self.init_kwargs)
        else:
            self.model = self.algo.load(checkpoint_path, self.env, tensorboard_log=self.log_dir)

    def fit(self, epochs, total_timesteps=None, last_epoch=None, last_timestep=None):
        if total_timesteps is None:
            total_timesteps = self.fit_kwargs['total_timesteps']

        if last_epoch is None:
            last_epoch = 0
            last_timestep = 0

        eval_freq = max(self.fit_kwargs['eval_freq'] // self.n_envs, 1)
        n_eval_episodes = self.fit_kwargs['n_eval_episodes']
        callback = EvalCallback(self.test_env, best_model_save_path=self.checkpoints_dir,
                                log_path=self.log_dir, eval_freq=eval_freq,
                                n_eval_episodes=n_eval_episodes, deterministic=True,
                                render=False)

        print("\n-------------------------------")
        print(">>>    Start")
        print("-------------------------------\n")
        start_time = time.time()

        for epoch in range(epochs):
            current_epoch = (epoch+1)
            current_total_epoch = current_epoch + last_epoch
            try:
                print("\n-------------------------------")
                print(">>>    Start training epoch n°", current_total_epoch)
                print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
                print("[INFO]  >>    scene: ", self.env_id)
                print("[INFO]  >>    algo: ", self.algo_name)
                print("[INFO]  >>    seed: ", self.seed)
                print("-------------------------------\n")

                self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True, log_interval=1, tb_log_name="log", callback=callback)
                self.save(current_epoch, total_timesteps, current_total_epoch, last_timestep)
            except:
               print("[ERROR]  >> The simulation failed. Restart from previous id.")
               
        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    def fit_continue(self, epochs, total_timesteps=None):
        last_timestep = self.params['model_params']['last_timestep']
        last_epoch = self.params['model_params']['last_epoch']

        print(f">>>    Continue model training from timestep: {last_timestep}")
        self.fit(epochs, total_timesteps, last_epoch, last_timestep)
    
    def eval(self, n_episodes, **kwargs):
        model_timestep = kwargs.get('model_timestep', 'best_model')
        render = kwargs.get('render', False)
        record = kwargs.get('record', False)

        if render:
            config = {"render": 1}
        else:
            config = {"render": 0}
            if record:
                record = False
                warnings.warn("Video recording not possible if rendering is off, record set to False", UserWarning)
        
        eval_env = SubprocVecEnv([make_env(self.env_id, 0, self.seed*10, self.max_episode_steps, config=config)])
        eval_env = VecNormalize.load(self.stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env = VecMonitor(eval_env, self.log_dir)

        if record:
            eval_env = VecVideoRecorder(eval_env, self.video_dir,
                                        record_video_trigger=lambda x: x == 0, video_length=self.max_episode_steps,
                                        name_prefix=self.model_name)
            
        checkpoint_path = f"{self.checkpoints_dir}/{model_timestep}"
        test_model = self.algo.load(checkpoint_path, eval_env, tensorboard_log=self.log_dir)

        if test_model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.env, eval_env)
            except AttributeError as error:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from error

        print("\n-------------------------------")
        print(f">>>    Testing model at timestep: {model_timestep}")

        mean_reward, std_reward = evaluate_policy(test_model, eval_env, n_eval_episodes=n_episodes, deterministic=True, render=render)
        print(f"Reward: {mean_reward}, Standard Devation: {std_reward}")
        
        print(">>   End.")

    def save(self, current_epoch, total_timesteps, current_total_epoch, last_timestep):
        current_timestep = (total_timesteps*current_epoch) + last_timestep

        self.model.save(f"{self.checkpoints_dir}/{current_timestep}")
        self.model.save(f"{self.checkpoints_dir}/latest")
        self.env.save(self.stats_path)

        model_log = self.params.copy()
        
        model_params = dict(env_id=self.env_id, algo=self.algo_name, seed=self.seed, last_epoch=current_total_epoch, last_timestep=current_timestep, n_envs=self.n_envs, model_name=self.model_name)
        model_log['model_params'] = model_params

        with open(self.model_log_path, 'wb') as model_log_file:
            pickle.dump(model_log, model_log_file)
    
    @classmethod
    def load(cls, model_dir, model_timestep='latest'):
        checkpoint_path = f"{model_dir}/model/{model_timestep}"
        if not os.path.exists(checkpoint_path + ".zip"):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model file does not exist")
            exit(1)
        
        output_dir = list(filter(None, model_dir.split("/")))[-4]
        log_dir = f"{model_dir}/log"

        model_log_path = f"{log_dir}/model_log.pkl"
        if not os.path.exists(model_log_path):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model log file does not exist")
            exit(1)

        with open(model_log_path, 'rb') as model_log_file:
            model_log = pickle.load(model_log_file) 
        
        model_params = model_log['model_params']
        env_id = model_params['env_id']
        algo_name = model_params['algo']
        seed = model_params['seed']
        n_envs = model_params['n_envs']
        max_episode_steps = model_log['fit_kwargs']['max_episode_steps']
        model_name = model_params['model_name']

        agent = cls(env_id, algo_name, seed, output_dir, max_episode_steps, n_envs, model_name, **model_log)

        return agent
        
    def env_wrap(self, n_envs, normalize=True):
        vec_env = SubprocVecEnv([make_env(self.env_id, i, self.seed, self.max_episode_steps) for i in range(n_envs)])
        self.test_env = SubprocVecEnv([make_env(self.env_id, 0, self.seed*10, self.max_episode_steps)])
        
        if normalize:
            if not os.path.exists(self.stats_path):
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
                self.test_env = VecNormalize(self.test_env, norm_obs=True, training=False, norm_reward=False)
            else:
                vec_env = VecNormalize.load(self.stats_path, vec_env)
                self.test_env = VecNormalize.load(self.stats_path, self.test_env)
                self.test_env.training = False
                self.test_env.norm_reward = False

        vec_env = VecMonitor(vec_env, self.log_dir)
        self.test_env = VecMonitor(self.test_env, self.log_dir)

        return vec_env

    def policy(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
