# -*- coding: utf-8 -*-
"""Funcitons to create and manage a server that distributes the computations
to its clients.
"""

__authors__ = ("PSC", "dmarchal", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "damien.marchal@univ-lille.fr", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, CNRS, Inria"
__date__ = "Oct 7 2020"


from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn

import socketserver
import threading
import subprocess
import queue
from os.path import dirname, abspath
import copy
import time

path = dirname(dirname(abspath(__file__))) + '/'


class SimpleThreadedXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

    def log_message(self, format, *args):
        pass


class CustomQueue(queue.Queue):
    """System to save element with a blocking get.

    Methods:
    -------
        __init__: Initialization of all arguments.
        __str__: Returns elements in string format.
        put: add element in the queue.
        get: remove and return element from the queue.
        front: first element in the queue.
        back; last element in the queue.

    Arguments:
    ---------
        See queue.Queue for all arguments.
        entries: list
            List to save elements.

    """
    def __init__(self):
        """Initialization of all arguments.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        queue.Queue.__init__(self)
        self.entries = []

    def __str__(self):
        """Returns elements in string format.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        return str(self.entries)

    def put(self, item):
        """Add one element in the queue.

        Parameters:
        ----------
            item:
                The element to add in the queue.

        Returns:
        -------
            None.
        """
        self.entries.append(item)
        queue.Queue.put(self, item)

    def get(self, timeout=None):
        """Remove and return the head of the queue.

        Parameters:
        ----------
            timeout: int or None, default = None
                Avoid blocking situations.

        Returns:
        -------
            res:
                The head of the queue.

        """
        res = queue.Queue.get(self, timeout=timeout)
        self.entries.pop(0)
        return res

    def front(self):
        """Get the first element of the queue.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The first element of the queue.

        """
        return self.entries[0]

    def back(self):
        """Get the last element of the queue.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The last element of the queue.

        """
        return self.entries[-1]

    def __len__(self):
        """Get the size of the queue.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The size of the queue.

        """
        return len(self.entries)


# Dictionary indexed by process ids containing job queues.
deterministic = True
stateId = 0
actions_to_stateId = {}
instances = {}
firstObservation = CustomQueue()
results = {}
planning = None

port_rpc = None


def get_id_from_actions(actions):
    """Create an id for a sequence of actions.

    If the sequence of actions is not associated with an id, create a new id and
    the saving elements associated with this id.

    Parameters:
    ----------
        actions: list
            The sequence of actions.

    Returns:
    -------
        The id (of the client) corresponding to the sequence of actions.

    """
    global stateId, instances
    k = str(actions)

    if k not in actions_to_stateId:
        stateId += 1
        actions_to_stateId[k] = [stateId]
        instances[stateId] = {"pendingTasks": CustomQueue(),
                              "pendingResults": CustomQueue(),
                              "positionResults": CustomQueue()}

    return actions_to_stateId[str(actions)][0]


def get_new_id(actions):
    """Add new id to a sequence of actions.

    If this sequence of actions is not associated with an id, create all the
    elements.

    Parameters:
    ----------
        actions: list
            The sequence of actions.

    Returns:
    -------
        The new id (of the new client) corresponding to the sequence of actions.

    """
    global stateId, instances

    stateId += 1
    if actions_to_stateId.get(str(actions)):
        actions_to_stateId[str(actions)] += [stateId]
    else:
        actions_to_stateId[str(actions)] = [stateId]
    instances[stateId] = {"pendingTasks": CustomQueue(),
                          "pendingResults": CustomQueue(),
                          "positionResults": CustomQueue()}

    return stateId


def registerFirstObservation(obs):
    """Function to save the first observation.

    Parameters:
    ----------
        obs:
            The observation.

    Returns:
    -------
        "ThankYou" to notify the connection.
    """

    global firstObservation
    firstObservation.put(obs)
    return "ThankYou"


# ################################## API RPC ################################## #

def registerInstance(state_id, process_id, history):
    """Update the dictionary associated with an id.

    Add information like list of actions and pid.

    Parameters:
    ----------
        state_id: int
            The id of the client.
        process_id: int
            The pid of the associated processus.
        history: list
            The sequence of actions.

    Returns:
    -------
        "ThankYou" to notify the connection.

    """
    global instances
    instances[state_id].update({"processId": process_id,
                               "history": history})

    return "ThankYou"


def getNextTask(state_id):
    """Distribute a pending task to the client.

    Parameters:
    ----------
        state_id: int
            The id of the client (associated with a sequence of actions).

    Returns:
    -------
        res:
            The pending task.
    """

    res = instances[state_id]["pendingTasks"].get()
    return res


def taskDone(state_id, history, result):
    """Notify the server that a submitted task has been terminated by the client.

    Parameters:
    ----------
        state_id: int
            The id of the client (associated with a sequence of actions).
        history: list
            The sequence of actions.
        result:
            The result of the task terminated by the client.

    Returns:
    -------
        "ThankYou" to notify the connection.

    """
    instances[state_id]["history"] = history
    instances[state_id]["pendingResults"].put(result)

    return "ThankYou"


def posDone(state_id, pos):
    """Notify the server that a position is send by the client.

    Parameters:
    ----------
        state_id: int
            The id of the client (associated with a sequence of actions).
        pos:
            The position returned by the client.

    Returns:
    -------
        "ThankYou" to notify the connection.

    """
    instances[state_id]["positionResults"].put(pos)

    return "ThankYou"


# #################### API python (on the environment side) #################### #

def make_action(command, **kwargs):
    """Add a command in the parameters.

    Parameters:
    ----------
        command:
            The command to add.
        kwargs: Dictionary
            Additional arguments.

    Returns:
    -------
        m: Dictionary
            The updated arguments.

    """
    m = {"command": command}
    m.update(kwargs)
    return m


def avalaible_port(to_str=False):
    """Find a free port to connect a server.

    Parameters:
    ----------
        to_str: bool
            Choose if the returns is an int or str.

    Returns:
    -------
        free_port: int or str
            The num of the port.
    """
    with socketserver.TCPServer(("localhost", 0), None) as s:
        free_port = s.server_address[1]
        print(free_port)

    if to_str:
        return str(free_port)
    else:
        return free_port


def start_server(config):
    """Start new server & first client in two dedicated threads.

       This function is not blocking and does not returns any values.
       Once the server is started it is possible to submit new tasks unsing
       the add_new_step function.It is then possible to get the results of the
       tasks using the get_results functions.

    Parameters:
    ----------
        config: dic
            Configuration of the environment.

    Returns:
    -------
            _: dic
                The port of the rpc server.

    """
    global planning, port_rpc

    planning = config['planning']

    if port_rpc is None:
         port_rpc= avalaible_port()

    # Register functions
    def dispatch(port_rpc):
        with SimpleThreadedXMLRPCServer(('localhost', port_rpc), requestHandler=RequestHandler) as server:
            server.register_function(getNextTask)
            server.register_function(taskDone)
            server.register_function(posDone)
            server.register_function(registerInstance)
            server.register_function(registerFirstObservation)
            server.serve_forever()

    global deterministic
    deterministic = config['deterministic']

    # Starts the server thread with the context.
    server_thread = threading.Thread(target=dispatch, args=(port_rpc,))
    server_thread.daemon = True
    server_thread.start()


def close_scene():
    """Ask the clients to close the scenes.

    Parameters:
    ----------
        None.

    Returns:
    -------
        None.
    """
    global stateId, actions_to_stateId, instances

    for instance in instances.values():
        instance["pendingTasks"].put(make_action("exit"))

    # Wait to close all clients
    time.sleep(0.01)


def clean_registry(history):
    """Close the clients of the useless branches.

    Usefull only in planning.

    Parameters:
    ----------
        history: list
            The sequence of actions.

    Returns:
    -------
        None.

    """
    global stateId, actions_to_stateId, instances
    id_closed = set()
    if len(history) > 0:
        copy_dic = copy.copy(instances)

        for instance in copy_dic.values():
            for i, action in enumerate(history):

                if instance.get('history'):
                    if len(instance['history']) < len(history) or instance['history'][i] != action:
                        id = get_id_from_actions(instance['history'])
                        if id not in id_closed:
                            try:
                                instances[id]["pendingTasks"].put(make_action("exit"))
                                id_closed.add(id)
                            except KeyError:
                                print("KeyError ", get_id_from_actions(instance['history']))
                                pass

    for id in id_closed:
        # while instances[id]["pendingTasks"].__len__()!=0:
        #     pass
        actions_to_stateId.pop(str(instances[id]['history']))
        instances.pop(id)


def start_scene(config, nb_actions):
    """Start the first client.

    Parameters:
    ----------
        config: Dictionary
            The configuration of the environment.
        nb_action: int
            The number of actions in the environment.

    Returns:
    -------
        obs:
            The first observation.

    """
    global stateId, actions_to_stateId, instances, firstObservation, results, port_rpc

    close_scene()

    # Information of the first client
    stateId = 0
    actions_to_stateId = {'[]': [0]}
    instances = {0: {"history": '[]',
                     "pendingTasks": CustomQueue(),
                     "pendingResults": CustomQueue(),
                     "positionResults": CustomQueue()}}
    firstObservation = CustomQueue()

    results = {}

    # Run the first client
    def deferredStart():
        sdict = str(config)
        subprocess.run([config['python_version'], path+"common/rpc_client.py", sdict, str(nb_actions), str(port_rpc)],
                       check=True)

    first_worker_thread = threading.Thread(target=deferredStart)
    first_worker_thread.daemon = True
    first_worker_thread.start()

    return firstObservation.get()


def add_new_step(history, new_action):
    """Ask a client to calculate the result for a given sequence of actions.

    Parameters:
    ----------
        history: list
            The sequence of past actions.
        new_action: int
            The new action.

    Returns:
    -------
        nid: int
            The id of the client that calculate the result.

    Note:
    ----
        If we don't realise planning, no fork.

    """
    global instances, actions_to_stateId, planning
    id = get_id_from_actions(history)

    if planning:
        if deterministic and str(history+[new_action]) in actions_to_stateId:
            # Transition has already been simulated
            return get_id_from_actions(history+[new_action])

        if deterministic:
            nid = get_id_from_actions(history+[new_action])
        else:
            nid = get_new_id(history+[new_action])
        instances[id]["pendingTasks"].put(make_action("fork_and_animate", stateId=nid, action=new_action))
    else:
        nid = id
        actions_to_stateId = {str(history + [new_action]): [id]}
        instances[id]["pendingTasks"].put(make_action("animate", action=new_action))

    return nid


def get_result(result_id, timeout=None):
    """Returns available results. Blocks until a result is available.

    Parameters:
    ----------
        result_id: int
            The id of the client who has to give his result.
        timeout: int or None, default = None
            To avoid blocking situations.

    Returns:
    -------
        The results.

    """
    global instances, deterministic, planning

    if deterministic and planning:
        if results.get(result_id):
            return results[result_id]
        else:
            try:
                res = instances[result_id]["pendingResults"].get(timeout=timeout)
            except queue.Empty:
                print("TIMEOUT ", timeout)
                res = {"stateId": result_id,
                       "observation": "",
                       "reward": 0.0,
                       "done": True,
                       "info": {"error": "TIMEOUT"}
                       }
            results[result_id] = res
            return res

    return instances[result_id]["pendingResults"].get(timeout=timeout)


def get_position(actions, timeout=None):
    """Returns available results. Blocks until a result is available.

    Parameters:
    ----------
        actions: list
            The sequence of past actions.
        timeout: int or None, default = None
            To avoid blocking situations.

    Returns:
    -------
        The results.

    """

    id = get_id_from_actions(actions)
    instances[id]["pendingTasks"].put(make_action("get_position"))
    try:
        pos = instances[id]["positionResults"].get(timeout=timeout)
    except queue.Empty:
        print("TIMEOUT ", timeout)
        pos = {"position": []}

    return pos
