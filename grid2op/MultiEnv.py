"""
This class allows to evaluate a single agent instance on multiple environments running in parrallel.

It uses the python "multiprocessing" framework to work, and thus is suitable only on a single machine.

More MlutiEnvironment class might come in the future.

It is a similar approach for the gym SubprocVecEnv class.
"""



import copy
import os
import time
from multiprocessing import Process, Pipe
import numpy as np

# try:
#     import gym
#     can_use = True
# except (ImportError, ModuleNotFoundError):
#     can_use = False

try:
    from .Exceptions import *
    from .Space import GridObjects
    from .Environment import Environment

except (ImportError, ModuleNotFoundError):
    from Exceptions import *
    from Space import GridObjects
    from Environment import Environment

    from Agent import DoNothingAgent
    from MakeEnv import make


import pdb


class Test(object):
    def __init__(self, i):
        self.i_ = i

    def step(self):
        tmp = 0
        for _ in range(10000):
            tmp += 1
        return self.i_


def worker(remote, parent_remote, params):
    parent_remote.close()
    # env = env_fn_wrapper.x()
    backend = params["backendClass"]()
    del params["backendClass"]
    env = Environment(**params, backend=backend)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class MultiEnvironment(GridObjects):
    def __init__(self, nb_env, env):
        GridObjects.__init__(self)
        # self.init_grid(env)
        self.imported_env = env
        self.nb_env = nb_env

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nb_env)])

        env_params = [env.get_kwargs() for _ in range(self.nb_env)]
        for el in env_params:
            el["backendClass"] = type(env.backend)
        self.ps = [Process(target=worker, args=(work_remote, remote, env_))
                   for (work_remote, remote, env_) in zip(self.work_remotes, self.remotes, env_params)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))

        observation_space, action_space = self.remotes[0].recv()
        self.waiting = True

    def send_act(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def wait_for_obs(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos


if __name__ == "__main__":
    env = make()
    nb_env = 5

    agent = DoNothingAgent(env.action_space)
    multi_envs = MultiEnvironment(env=env, nb_env=nb_env)

    multi_envs.send_act([agent.act(None, None, None) for _ in range(5)])
    obs, rewards, dones, info = multi_envs.wait_for_obs()
    pdb.set_trace()




