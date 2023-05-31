from abc import abstractmethod

from agents.SofaTestAgent import SofaTestAgent


class SofaBaseAgent(SofaTestAgent):
    def __init__(self, env_id, seed=None, output_dir=None, max_episode_steps=None):
        super().__init__(env_id, seed, output_dir, max_episode_steps)
        
        self.env = self.env_make(max_episode_steps)

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def fit_continue(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, **kwargs):
        pass

    def close(self):
        self.env.close()
        self.test_env.close()
