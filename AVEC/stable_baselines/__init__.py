from stable_baselines.ppo2 import PPO2
from stable_baselines.sac import SAC

# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.trpo_mpi import TRPO
del mpi4py

__version__ = "2.9.0a0"
