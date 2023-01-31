from sofagym.envs.BubbleMotion.BubbleMotionEnv import *
from sofagym.envs.CartStem.CartStemEnv import *
from sofagym.envs.CartStemContact.CartStemContactEnv import *
from sofagym.envs.CatchTheObject.CatchTheObjectEnv import *
from sofagym.envs.CTR.CTREnv import *
from sofagym.envs.Diamond.DiamondEnv import *
from sofagym.envs.Gripper.GripperEnv import *
from sofagym.envs.Maze.MazeEnv import *
from sofagym.envs.MultiGaitRobot.MultiGaitRobotEnv import *
from sofagym.envs.SimpleMaze.SimpleMazeEnv import *
from sofagym.envs.StemPendulum.StemPendulumEnv import *
from sofagym.envs.Trunk.TrunkEnv import *
from sofagym.envs.TrunkCup.TrunkCupEnv import *


# registering sofagym envs as gymnasium envs
from gym.envs.registration import register
register(
    id='bubblemotion-v0',
    entry_point='sofagym.envs:BubbleMotionEnv',
)
register(
    id='cartstem-v0',
    entry_point='sofagym.envs:CartStemEnv',
)
register(
    id='catchtheobject-v0',
    entry_point='sofagym.envs:CatchTheObject',
)
register(
    id='cartstemcontact-v0',
    entry_point='sofagym.envs:CartStemContactEnv',
)
register(
    id='concentrictuberobot-v0',
    entry_point='sofagym.envs:ConcentricTubeRobotEnv',
)
register(
    id='diamondrobot-v0',
    entry_point='sofagym.envs:DiamondRobotEnv',
)
register(
    id='gripper-v0',
    entry_point='sofagym.envs:GripperEnv',
)
register(
    id='maze-v0',
    entry_point='sofagym.envs:MazeEnv',
)
register(
    id='multigaitrobot-v0',
    entry_point='sofagym.envs:MultiGaitRobotEnv',
)
register(
    id='simple_maze-v0',
    entry_point='sofagym.envs:SimpleMazeEnv',
)
register(
    id='stempendulum-v0',
    entry_point='sofagym.envs:StemPendulumEnv',
)
register(
    id='trunk-v0',
    entry_point='sofagym.envs:TrunkEnv',
)
register(
    id='trunkcup-v0',
    entry_point='sofagym.envs:TrunkCupEnv',
)
