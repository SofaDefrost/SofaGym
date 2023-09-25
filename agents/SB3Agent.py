import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

import torch as th
import torch.nn as nn
import yaml
from agents.callbacks import (SaveBestNormalizeCallback, SaveCallback,
                              VideoRecordCallback)
from agents.SofaBaseAgent import SofaBaseAgent
from agents.utils import make_env, mkdirp, sec_to_hours
from colorama import Fore
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import (SubprocVecEnv,
                                              VecMonitor, VecVideoRecorder,
                                              sync_envs_normalization)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


# Adapted from RL Baselines3 Zoo
def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess scheduled training hyperparameters.
    
    Parameters
    ----------
    hyperparams : Dict[str, Any]
        A dictionary containing the training algorithm hyperparameters.
    
    Returns
    -------
    hyperparams : Dict[str, Any]
       The hyperparams dictionary after preprocessing the schedules.
    """
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
    """Create a schedule that computes the current hyperparameter value 
    based on the remaining progress.
    
    Parameters
    ----------
    initial_value : float
        The initial hyperparameter value.
    
    Returns
    -------
    func : Callable[[float], float]
        A callable object, which is a schedule function that takes a single argument 
        `progress_remaining` and returns a float value representing the current
        hyperparameter value.
    """
    def func(progress_remaining: float) -> float:
        """Calculate the current hyperparameter value based on the remaining progress.
        
        Parameters
        ----------
        progress_remaining: float
            The remaining progress of the training task, ranging from 
            1 (beginning) to 0 (completed).
        
        Returns
        -------
        float
            the current hyperparameter value.
        """
        return progress_remaining * initial_value

    return func


class SB3Agent(SofaBaseAgent):
    """Class for creating, training, and evaluating reinforcement learning policies in a SOFA scene 
    with gym environment using Stable Baselines3.

    Inherits from the `SofaBaseAgent` class.

    Parameters
    ----------
    env_id : str
        The name of the environment to be used.
    algo_name: str
        The name of the RL algorithm to be used for training.
    seed : int, default=0
        The seed used for random number generation to initialize the environment.
    output_dir : str, default="./Results"
        The output directory to save the environment results.
    max_episode_steps : int, default=None
        The maximum number of steps that an agent can take in an episode of the environment.
        Once this limit is reached, the episode is terminated. If None, the episode will 
        continue until the goal is done or a termination conidition happens.
    n_envs: int, default=1
        The number of environment instances created simultaneously to train the agent on 
        multiple environments in parallel.
    model_name : str, default=None
        The name of the training model. Used for loading a previously trained model.
    
    Attributes
    ----------
    env_id : str
        The name of the environment to be used.
    seed : int
        The seed used for random number generation to initialize the environment.
    output_dir : str
        The output directory to save the environment results.
    max_episode_steps : int
        The maximum number of steps that an agent can take in an episode of the environment.
        Once this limit is reached, the episode is terminated. If None, the episode will 
        continue until the goal is done or a termination conidition happens.
    n_envs : int
        The number of environment instances created simultaneously to train the agent on 
        multiple environments in parallel.
    test_env : gym.Env
        The test environment instance.
    env : gym.Env
        The training environment instance.
    algo_name : str
        The name of the RL algorithm to be used for training.
    algo : OnPolicyAlgorithm
        The RL algorithm instance.
    params : dict
        The parameters of the RL algorithm and the model.
    
    Notes
    -----
    This class inherits from the abstract class `SofaBaseAgent` and override the abstract methods defined.

    Usage:
    -----
    - Use the `fit` method to train a new model.
    - Use the `eval` method to evaluate the performance of a trained model.
    - Use the `load` method to load a previously trained model to continue training using `fit` or evaluate it using `eval`.
    """
    def __init__(self, env_id, algo_name, seed=0, output_dir="./Results", max_episode_steps=None, n_envs=1, model_name=None, **kwargs):
        """Initialization of stable baselines3 agent class. Creates the environment for the SOFA scene, 
        which will be used for the training and evaluation using RL.
        
        Parameters
        ----------
        env_id : str
            The name of the environment to be used.
        algo_name : str
            The name of the RL algorithm to be used for training.
        seed : int, default=0
            The seed used for random number generation to initialize the environment.
        output_dir : str, default="./Results"
            The output directory to save the environment results.
        max_episode_steps : int, default=None
            The maximum number of steps that an agent can take in an episode of the environment.
            Once this limit is reached, the episode is terminated. If None, the episode will 
            continue until the goal is done or a termination condition happens.
        n_envs : int, default=1
            The number of environment instances created simultaneously to train the agent on 
            multiple environments in parallel.
        model_name : str, default=None
            The name of the training model. Used for loading a previously trained model.
        """
        super().__init__(env_id, seed, output_dir, max_episode_steps, n_envs)

        self.algo_name = algo_name
        self.algo = eval(self.algo_name)
        self.model_name = model_name
        self.params = kwargs

        self.init_model()

    def init_dirs(self):
        """Initialize directories for logging data andsaving the model's training checkpoints
        and videos.
        """
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

        self.model_log_path = f"{self.checkpoints_dir}/model_log.pkl"

    def create_model_log(self):
        """Create a log for model parameters and save it as a pickle file.
        """
        model_log = self.params.copy()
        model_params = dict(env_id=self.env_id, algo=self.algo_name, seed=self.seed,
                            n_envs=self.n_envs, model_name=self.model_name)
        model_log['model_params'] = model_params

        with open(self.model_log_path, 'wb') as model_log_file:
            pickle.dump(model_log, model_log_file)

    def load_params(self):
        """Load hyperparameters and preprocesses them.
        """
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
        self.video_length = self.fit_kwargs['video_length']

        if self.max_episode_steps is None:
            self.max_episode_steps = self.fit_kwargs['max_episode_steps']

        # Check if a pre-trained agent is loaded and get loaded timestep
        if self.params.get('model_params'):
            self.model_timestep = self.params['model_params'].get('loaded_timestep')
        else:
            self.model_timestep = None
            self.create_model_log()
            
    def init_model(self):
        """Initialize the model.
        """
        self.init_dirs()
        self.load_params()

        self.env = self.env_wrap(self.n_envs, normalize=True)

        if not self.model_timestep:
            self.model = self.algo(env=self.env, seed=self.seed,
                                   verbose=1, tensorboard_log=self.log_dir,
                                   **self.init_kwargs)
        else:
            checkpoint_path = f"{self.checkpoints_dir}/{self.model_timestep}"
            self.model = self.algo.load(checkpoint_path, self.env, tensorboard_log=self.log_dir)

    def fit(self, total_timesteps=None):
        """Train the model.
        
        Parameters
        ----------
        total_timesteps: int, default=None
            The number of total_timesteps to train the model.
        
        Returns
        -------
        None.
        """
        if total_timesteps is None:
            total_timesteps = self.fit_kwargs['total_timesteps']

        save_freq = max(self.fit_kwargs['save_freq'] // self.n_envs, 1)
        eval_freq = max(self.fit_kwargs['eval_freq'] // self.n_envs, 1)
        n_eval_episodes = self.fit_kwargs['n_eval_episodes']

        save_callback = SaveCallback(save_freq=save_freq,
                                           save_path=self.checkpoints_dir,
                                           save_replay_buffer=True, save_vecnormalize=True,
                                           verbose=2)
        save_normalization = SaveBestNormalizeCallback(save_path=self.checkpoints_dir, verbose=2)
        video_record_callback = VideoRecordCallback(self.video_dir, self.video_length, self.log_dir, verbose=2)
        eval_callback = EvalCallback(self.test_env, best_model_save_path=self.checkpoints_dir,
                                     log_path=self.log_dir, eval_freq=eval_freq,
                                     n_eval_episodes=n_eval_episodes, deterministic=True,
                                     callback_on_new_best=save_normalization,
                                     callback_after_eval=video_record_callback,
                                     verbose=1,
                                     render=False)

        print("\n-------------------------------")
        print(">>>    Start")
        print("-------------------------------\n")
        start_time = time.time()

        print("\n-------------------------------")
        print(">>>    Start training")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
        print("[INFO]  >>    scene: ", self.env_id)
        print("[INFO]  >>    algo: ", self.algo_name)
        print("[INFO]  >>    seed: ", self.seed)
        print("-------------------------------\n")

        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False,
                         progress_bar=True, log_interval=1, tb_log_name="log",
                         callback=[eval_callback, save_callback])

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    def eval(self, n_episodes, **kwargs):
        """Evaluate the trained model's performance.
        
        Parameters
        ----------
        n_episodes: int
            The number of evaluation episodes for the model.
        
        Returns
        -------
        None.
        """
        model_timestep = kwargs.get('model_timestep', 'best_model')
        render = kwargs.get('render', False)
        record = kwargs.get('record', False)

        checkpoint_path = f"{self.checkpoints_dir}/{model_timestep}"
        eval_model = self.algo.load(checkpoint_path, self.test_env, tensorboard_log=self.log_dir)

        if eval_model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.env, self.test_env)
            except AttributeError as error:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from error

        print("\n-------------------------------")
        print(f">>>    Testing model at timestep: {model_timestep}")

        mean_reward, std_reward = evaluate_policy(eval_model, self.test_env,
                                                  n_eval_episodes=n_episodes,
                                                  deterministic=True, render=False)

        if record and not render:
            render = True
            warnings.warn("Video recording not possible if rendering is off, render set to True", UserWarning)

        if render:
            config = {"render": 1}
            best_model_vecnormalize_path = f"{self.checkpoints_dir}/vecnormalize_best_model.pkl"
            eval_env = SubprocVecEnv([make_env(self.env_id, 0, self.seed, self.max_episode_steps, config=config)])
            eval_env = VecNormalize.load(best_model_vecnormalize_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            if record:
                eval_env = VecVideoRecorder(eval_env, self.video_dir,
                                            record_video_trigger=lambda x: x == 0,
                                            video_length=self.video_length,
                                            name_prefix="eval_video")
            eval_env = VecMonitor(eval_env, self.log_dir)

            eval_model = self.algo.load(checkpoint_path, eval_env, tensorboard_log=self.log_dir)

            if eval_model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.env, eval_env)
                except AttributeError as error:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from error

            _, _ = evaluate_policy(eval_model, eval_env, n_eval_episodes=1,
                                   deterministic=True, render=True)

            eval_env.close_video_recorder()
            eval_env.close()

        print(f"Reward: {mean_reward}, Standard Devation: {std_reward}")
        print(">>   End.")

    @classmethod
    def load(cls, model_dir, model_timestep='latest_model'):
        """Load a pre-trained model.
        
        Parameters
        ----------
        model_dir: str
            The directory where the model is saved.
        model_timestep: str, default='latest_model'
            The checkpoint of the model to load.
        
        Returns
        -------
        agent : object
            The loaded Stable Baselines3 agent with the pre-trained model.
        """
        checkpoint_dir = f"{model_dir}/model"
        checkpoint_path = f"{checkpoint_dir}/{model_timestep}"
        if not os.path.exists(checkpoint_path + ".zip"):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model file does not exist")
            exit(1)

        output_dir = list(filter(None, model_dir.split("/")))[-4]

        model_log_path = f"{checkpoint_dir}/model_log.pkl"
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

        model_log['model_params']['loaded_timestep'] = model_timestep

        agent = cls(env_id, algo_name, seed, output_dir, max_episode_steps, n_envs, model_name, **model_log)

        return agent
    
    def env_wrap(self, n_envs, normalize=True):
        """Create wrapped training and testing environments.
        
        Parameters
        ----------
        n_envs: int
            Number of training environments.
        normalize: bool, default=True
            Whether to wrap the environment to normalize the observations and rewards or not.
        
        Returns
        -------
        vec_env : object
            The wrapped training environment.
        """
        vec_env = SubprocVecEnv([make_env(self.env_id, i, self.seed, self.max_episode_steps) for i in range(n_envs)])
        self.test_env = SubprocVecEnv([make_env(self.env_id, 0, self.seed, self.max_episode_steps, config={"render": 1})])
        
        if normalize:
            if self.model_timestep is None:
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
                self.test_env = VecNormalize(self.test_env, norm_obs=True, training=False, norm_reward=False)
            else:
                vecnormalize_path = f"{self.checkpoints_dir}/vecnormalize_{self.model_timestep}.pkl"
                if not os.path.exists(vecnormalize_path):
                    print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model VecNormalize file does not exist")
                    exit(1)
                vec_env = VecNormalize.load(vecnormalize_path, vec_env)
                self.test_env = VecNormalize.load(vecnormalize_path, self.test_env)
                self.test_env.training = False
                self.test_env.norm_reward = False

        vec_env = VecMonitor(vec_env, self.log_dir)
        self.test_env = VecMonitor(self.test_env, self.log_dir)

        return vec_env

    def policy(self, obs, deterministic=True):
        """Perform prediction using the trained model
        
        Parameters
        ----------
        obs: object
            The current state of the environment.
        deterministic: bool, default=True
            Whether or not to return deterministic actions.
        
        Returns
        -------
        object
            The model's predicted action.
        """
        return self.model.predict(obs, deterministic=deterministic)
    
    def close(self):
        """Overrides `close` from `SofaTestAgent` class.
        Close the training and testing environments.
        """
        self.env.close()
        self.test_env.close()
