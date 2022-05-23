# -*- coding: utf-8 -*-
"""Client to run sofa scene.
"""

__authors__ = ("PSC", "dmarchal", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "damien.marchal@univ-lille.fr", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, CNRS, Inria"
__date__ = "Oct 7 2020"

import xmlrpc.client

import os
import sys
import ast
import json
import time
import random
import importlib

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../../")

from sofagym.env.common.simulate import init_simulation, step_simulation

stateId = 0
command = None
history = []
config = None
position = None


def do_fork(state_id, new_history):
    """Fork the processus.

    Parameters:
    ----------
        state_id:
            The id of the client in the son processus.
        new_history:
            The history of the client in the son processus.

    Returns:
    -------
        pid:
            The pid of son.
    """
    global stateId
    pid = os.fork()
    if pid == 0:
        stateId = state_id
        s.registerInstance(stateId, os.getpid(), new_history)
    return pid


def do_animate(task, _getReward_, _getState_, _startCmd, _getPos):
    """Compute one simulation step.

    Parameters:
    ----------
        task: Dictionary
            The task to do, contains the field "action".
        _getReward: function
            Allow to compute the reward.
        _getState: function
            Allow to compute the state.
        _startCmd: function
            Initialize the command.
        _getPos: function
            Get the position of the object in the scene.

    Returns:
    -------
        None.

    """
    global stateId, config, path, position

    scene = config['scene']
    save_data = config['save_data']

    action = task["action"]

    state = _getState(root)

    # Run simulation and get results (position, state, reward)
    pos = step_simulation(root, config, action, _startCmd, _getPos)

    position = pos
    obs = _getState(root)
    done, reward = _getReward(root)

    if config['save_data']:
        data = {"history": history,
                "configuration": config,
                "state": state,
                "action": action,
                "obs": obs}
        filename = config['save_path_results'] + "/" + scene + str(history) + ".json"
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    # Notify the server that the task is done
    s.taskDone(stateId, history, make_result(stateId, obs, reward, done))


def send_position():
    """Send position to the server.

    Parameters:
    ----------
        None.

    Returns:
    -------
        None.

    """
    global stateId, position

    if position is not None:
        s.posDone(stateId, {"position": position})
    else:
        s.posDone(stateId, {"position": []})


def make_result(state_id, obs, reward, done):
    """Put the result in the right format.

    Parameters:
    ----------
        state_id: int
            The id of the client.
        obs:
            The new state.
        reward:
            The reward obtains after applying action in old state.
        done: bool
            Whether or not the goal is reached.

    Returns:
    -------
        A dictionary containing the result in the right format.

    """

    return {"stateId": state_id,
            "observation": obs,
            "reward": float(reward),
            "done": done,
            "info": {}}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("SYNTAX: python3.7 client.py config nb_actions port_rpc")
        sys.exit(-1)

    config = ast.literal_eval(sys.argv[1])
    nb_actions = sys.argv[2]
    port_rpc = sys.argv[3]

    scene = config['scene']

    # Register the first instance to the manager
    s = xmlrpc.client.ServerProxy('http://localhost:' + port_rpc)
    s.registerInstance(0, os.getpid(), [])

    _getState = importlib.import_module("sofagym.env."+scene+"."+scene+"Toolbox").getState
    _getReward = importlib.import_module("sofagym.env."+scene+"."+scene+"Toolbox").getReward
    _startCmd = importlib.import_module("sofagym.env."+scene+"."+scene+"Toolbox").startCmd
    _getPos = importlib.import_module("sofagym.env."+scene+"."+scene+"Toolbox").getPos

    root = init_simulation(config, _startCmd, mode='simu')

    s.registerFirstObservation(make_result(stateId, _getState(root), 0.0, False))

    command = None

    while command != "exit":
        task = s.getNextTask(stateId)
        command = task["command"]
        if command == "fork":
            do_fork(task["stateId"], history)

        elif command == "animate":
            history.append(task["action"])
            do_animate(task, _getReward, _getState, _startCmd, _getPos)

        elif command == "fork_and_animate":
            action = task["action"]
            pid = do_fork(task["stateId"], history + [task["action"]])
            if pid == 0:
                history.append(action)
                do_animate(task, _getReward, _getState, _startCmd, _getPos)

        elif command == "get_position":
            send_position()

    # Avoid that all client exit in the same time
    time.sleep(random.random())
