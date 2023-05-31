import random
from abc import ABC
from statistics import mean
from typing import Optional

import gym
import numpy as np
import torch as th
from agents.utils import mkdirp


class SofaTestAgent(ABC):
    def __init__(
        self,
        env_id: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
    ):
        self.env_id = env_id
        self.seed = seed
        self.output_dir = output_dir

        self.test_env = self.env_make(max_episode_steps)

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, seed):
        if seed is None:
            seed = 0
            
        self._seed = seed
        random.seed(self._seed)
        th.manual_seed(self._seed)
        np.random.seed(self._seed)

    @property
    def output_dir(self):
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, output_dir):
        if output_dir is None:
            output_dir = "./Results"
        
        self._output_dir = output_dir
        mkdirp(self._output_dir)

    def env_make(self, max_episode_steps=None):
        env = gym.make(self.env_id)
        
        if max_episode_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps)
        
        self.env_seed(env)

        return env
    
    def env_seed(self, env):
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.action_space.np_random.seed(self.seed)

    def policy(self, obs):
        return self.test_env.action_space.sample(), obs

    def test(self, n_episodes=1, render=False):
        if render:
            self.test_env.config.update({"render":1})

        r = []

        for t in range(n_episodes):
            print("Start >> Episode", t+1)
            obs = self.test_env.reset()

            if render:
                self.test_env.render()

            rewards = []
            done = False
            id = 0

            while not done:
                action, _states = self.policy(obs)
                obs, reward, done, info = self.test_env.step(action)
                
                if render:
                    print("Episode", t+1, "- Step ", id+1 ,"- Took action: ", action, "- Current State: ", obs, "- Got reward: ", reward, "- Done: ", done)
                    self.test_env.render()
                
                rewards.append(reward)
                id +=1

            print("Done >> Episode", t+1, "- Reward = ", rewards, "- Sum reward:", sum(rewards))
            r.append(sum(rewards))

        print("[INFO]  >> Mean reward: ", mean(r))

        self.test_env.close()

        return mean(r)

    def close(self):
        self.test_env.close()
