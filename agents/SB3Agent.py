import os
import pickle
import time
from pathlib import Path

import torch as th
import yaml
from agents.SofaBaseAgent import SofaBaseAgent
from agents.utils import make_env, mkdirp, sec_to_hours
from colorama import Fore
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SB3Agent(SofaBaseAgent):
    def __init__(self, env_id, algo_name, seed=None, output_dir=None, max_episode_steps=None, n_envs=1, model_name=None, **kwargs):
        super().__init__(env_id, seed, output_dir, max_episode_steps)
        
        self.algo_name = algo_name
        self.algo = eval(self.algo_name)
        self.max_episode_steps = max_episode_steps
        self.n_envs = n_envs
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

        self.fit_kwargs = self.params['fit_kwargs']

        if self.max_episode_steps is None:
            self.max_episode_steps = self.fit_kwargs['max_episode_steps']
    
    def init_model(self):
        self.load_params()
        self.init_dirs()

        self.env = self.env_wrap(self.n_envs, normalize=True, max_episode_steps=self.max_episode_steps)
        
        self.model = self.algo(env=self.env, seed=self.seed, verbose=1, tensorboard_log=self.log_dir, **self.init_kwargs)

    def fit(self, epochs, total_timesteps=None, last_epoch=None, last_timestep=None):
        if total_timesteps is None:
            total_timesteps = self.fit_kwargs['total_timesteps']

        if last_epoch is None:
            last_epoch = 0
            last_timestep = 0

        best = -100000

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
            
                self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True, log_interval=1, tb_log_name="log")
                self.save(current_epoch, total_timesteps, current_total_epoch, last_timestep)

                print("\n-------------------------------")
                print(">>>    Start test epoch n°", current_total_epoch)
                print("[INFO]  >>    scene: ", self.env_id)
                print("[INFO]  >>    algo: ", self.algo_name)
                print("[INFO]  >>    seed: ", self.seed)
                print("-------------------------------\n")

                r = self.test(n_episodes=1, render=True)

                if r >= best:
                    print(">>>    Save training epoch n°", current_total_epoch)
                    self.model.save(f"{self.checkpoints_dir}/best")
                    best = r

            except:
               print("[ERROR]  >> The simulation failed. Restart from previous id.")
               
        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    def fit_continue(self, epochs, total_timesteps=None):
        last_timestep = self.params['model_params']['last_timestep']
        last_epoch = self.params['model_params']['last_epoch']

        print(f">>>    Continue model training from timestep: {last_timestep}") 
        self.fit(epochs, total_timesteps, last_epoch, last_timestep)
    
    def eval(self, n_episodes, model_timestep='best'):
        checkpoint_path = f"{self.checkpoints_dir}/{model_timestep}"
        self.model = self.algo.load(checkpoint_path, tensorboard_log=self.log_dir)
        self.model.set_env(self.env)

        print("\n-------------------------------")
        print(f">>>    Testing model at timestep: {model_timestep}")
        r = self.test(n_episodes=n_episodes, render=True)
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
        video_dir = f"{log_dir}/videos"
        stats_path = f"{log_dir}/vec_normalize.pkl" 

        model_log_path = f"{log_dir}/model_log.pkl"
        if not os.path.exists(model_log_path):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model log file does not exist")
            exit(1)

        with open(model_log_path, 'rb') as model_log_file:
            model_log = pickle.load(model_log_file) 
        
        model_params = model_log['model_params']
        env_id = model_params['env_id']
        algo_name = model_params['algo']
        algo = eval(algo_name)
        seed = model_params['seed']
        n_envs = model_params['n_envs']
        max_episode_steps = model_log['fit_kwargs']['max_episode_steps']
        model_name = model_params['model_name']

        agent = cls(env_id, algo_name, seed, output_dir, max_episode_steps, n_envs, model_name, **model_log)
        agent.env = VecNormalize.load(stats_path, agent.env)
        agent.model = algo.load(checkpoint_path, tensorboard_log=log_dir)
        agent.model.set_env(agent.env)

        return agent
        
    def env_wrap(self, n_envs, normalize=True, max_episode_steps=None):
        self.vec_env = SubprocVecEnv([make_env(self.env_id, i, self.seed, max_episode_steps) for i in range(n_envs)])
        
        if normalize:
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        self.vec_env = VecMonitor(self.vec_env, self.log_dir)

        self.test_env = make_env(self.env_id, 0, self.seed*10, max_episode_steps)()

        return self.vec_env

    def policy(self, obs, deterministic=False):
        return self.model.predict(obs, deterministic=deterministic)

    def close(self):
        self.env.close()
        self.test_env.close()
