import bz2
import os
import pickle
import time
from pathlib import Path

import _pickle as cPickle
import yaml
from agents.SofaBaseAgent import SofaBaseAgent
from agents.utils import sec_to_hours
from colorama import Fore
from rlberry.agents.experimental.torch import SACAgent
from rlberry.agents.torch import A2CAgent, DQNAgent, PPOAgent, REINFORCEAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from stable_baselines3.common.monitor import Monitor


class RLberryAgent(SofaBaseAgent):
    def __init__(self, env_id, algo_name, seed=0, output_dir="./Results", max_episode_steps=None, n_envs=1, checkpoint_file=None, **kwargs):
        super().__init__(env_id, seed, output_dir, max_episode_steps, n_envs)
        
        self.algo_name = algo_name
        self.algo = eval(self.algo_name + "Agent")
        self.checkpoint_file = checkpoint_file
        self.params = kwargs

        self.env = self.env_wrap()
        self.env_kwargs = None
        self.env_tuple = (self.env_ctor, self.env_kwargs)

        self.load_params()

        self.manager = self.create_agent()

    def load_params(self):
        if not self.params:
            self.params_path = "./agents/hyperparameters/rlberry_params.yml"
            config = yaml.safe_load(Path(self.params_path).read_text())
            self.params = config[self.algo_name]
        
        self.init_kwargs = self.params['init_kwargs']
        self.fit_kwargs = self.params['fit_kwargs']
        self.eval_kwargs = self.params['eval_kwargs']
        
    def create_agent(self):
        manager = AgentManager(
            self.SaveAgent(),
            self.env_tuple,
            init_kwargs=self.init_kwargs,
            fit_kwargs=self.fit_kwargs,
            eval_kwargs=self.eval_kwargs,
            n_fit=self.n_envs,
            parallelization='thread',
            enable_tensorboard=True,
            )
        
        return manager

    def fit(self, epochs, last_epoch=None):
        if last_epoch is None:
            last_epoch = 0

        print("\n-------------------------------")
        print(">>>    Start")
        print("-------------------------------\n")
        start_time = time.time()

        for epoch in range(epochs):
            current_epoch = (epoch+1)
            current_total_epoch = current_epoch + last_epoch
            
            print("\n-------------------------------")
            print(">>>    Start training epoch n°", current_total_epoch)
            print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
            print("[INFO]  >>    scene: ", self.env_id)
            print("[INFO]  >>    algo: ", self.algo_name)
            print("[INFO]  >>    seed: ", self.seed)
            print("-------------------------------\n")
        
            self.manager.fit()
            self.save(current_total_epoch)
            self.manager_file = self.manager.save()
            
        plot_writer_data(self.manager, tag='episode_rewards')

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    def fit_continue(self, epochs):
        last_epoch = self.params['model_params']['last_epoch']

        print(f">>>    Continue agent training from epoch {last_epoch} using checkpoint: {self.checkpoint_file}") 
        self.fit(epochs, last_epoch)
    
    def eval(self, n_tests):
        n_epochs = self.eval_kwargs['n_simulations']

        print("\n-------------------------------")
        print(f">>>    Testing agent")
        for test in range(n_tests):
            print("\n-------------------------------")
            print(">>>    Start test epoch n°", test+1)
            print("[INFO]  >>    scene: ", self.env_id)
            print("[INFO]  >>    algo: ", self.algo_name)
            print("[INFO]  >>    seed: ", self.seed)
            print("-------------------------------\n")

            _ = evaluate_agents(
                [self.manager], n_simulations=n_epochs, show=True
                )

    def save(self, current_total_epoch):
        model_log = self.params.copy()
        
        model_params = dict(env_id=self.env_id, algo=self.algo_name, seed=self.seed, last_epoch=current_total_epoch, n_envs=self.n_envs)
        model_log['model_params'] = model_params

        model_log_path = f"{self.manager.output_dir_}/model_log.pkl"
        with open(model_log_path, 'wb') as model_log_file:
            pickle.dump(model_log, model_log_file)
    
    @classmethod
    def load(cls, model_dir):
        checkpoint_file = f"{model_dir}/output_0/checkpoint.pickle"
        if not os.path.exists(checkpoint_file):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Agent checkpoint file does not exist")
            exit(1)
        
        with bz2.BZ2File(checkpoint_file, "rb") as file:
            data = cPickle.load(file)
            output_dir = str(data['kwargs']['output_dir'])

        model_log_path = f"{model_dir}/model_log.pkl"
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
        
        agent =  cls(env_id, algo_name, seed, output_dir, n_envs=n_envs, checkpoint_file=checkpoint_file, **model_log)
        
        manager_file = f"{model_dir}/manager_obj.pickle"
        if not os.path.exists(manager_file):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Mangager file does not exist")
            exit(1)
        
        agent.manager = AgentManager.load(manager_file)

        return agent

    def env_wrap(self):
        self.env_wrapped = Monitor(self.env)

        return self.env_wrapped
    
    def env_ctor(self):
        return self.env_wrap()

    def SaveAgent(self):
        parent = self
        algo = parent.algo
        env = parent.env_tuple

        global SaveAgent

        class SaveAgent(algo):
            def __init__(self, env, **kwargs):
                algo.__init__(self, env, **kwargs)

            def fit(self, budget: int):
                if parent.checkpoint_file is not None:
                    loaded_checkpoint = SaveAgent.load(parent.checkpoint_file, **self.get_params())
                    self.__dict__.update(loaded_checkpoint.__dict__)

                super().fit(budget)
                parent.checkpoint_file = self.save(self.output_dir / "checkpoint.pickle")
        
        return SaveAgent
