# Actor with Variance Estimated Critic (AVEC)

This is the implementation of Actor with Variance Estimated Critic (AVEC). The code is a fork of [Stable Baselines](https://github.com/hill-a/stable-baselines).

## Installation

This repository supports Tensorflow versions from 1.14.0 to 2.3.1. Works on GPUs.

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install the repository on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).


## Example

In order to use your RL algorithm with AVEC, simply use `avec_coef=1.` and `vf_coef=0.` (or `value_coef=0.` for SAC).

Run `python run.py` to start the training, monitor the learning logs (.csv, tensorboard) and reproduce the results of the paper with the environments and seeds of your choice.

Here is a quick example of how to train and run AVEC-PPO2 on a AntBullet environment:
```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

env = gym.make('AntBulletEnv-v0')

model = PPO2(MlpPolicy, env, verbose=1, avec_coef=1., vf_coef=0.)
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines import PPO2

model = PPO2('MlpPolicy', 'AntBulletEnv-v0', avec_coef=1., vf_coef=0.).learn(1000000)
```

Here is a quick example of how to train and run AVEC-TRPO on a AntBullet environment:
```python
from stable_baselines.trpo_mpi import TRPO

model = TRPO('MlpPolicy', 'AntBulletEnv-v0', avec_coef=1., vf_coef=0.).learn(1000000)
```

Finally, here is a quick example of how to train and run AVEC-SAC on a AntBullet environment:
```python
from stable_baselines.sac import SAC

model = SAC('CustomSACPolicy', 'AntBulletEnv-v0', avec_coef=1., value_coef=0.).learn(1000000)
```


Please read the [documentation](https://stable-baselines.readthedocs.io/) for more examples.


## MuJoCo
Some of the examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py).
