# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import numpy as np
import copy

from grid2op.dtypes import dt_int, dt_float
from grid2op.Space import GridObjects, RandomObject
from grid2op.Exceptions import EnvError, Grid2OpException


class MultiMixEnvironment(GridObjects, RandomObject):
    """
    This class represent a single powergrid configuration,
    backed by multiple environments parameters and chronics

    It implements most of the :class:`BaseEnv` public interface:
    so it can be used as a more classic environment.

    MultiMixEnvironment environments behave like a superset of the environment: they
    are made of sub environments (called mixes) that are grid2op regular :class:`Environment`.
    You might think the MultiMixEnvironment as a dictionary of :class:`Environment` that implements
    some of the :class:`BaseEnv` interface such as :func:`BaseEnv.step` or :func:`BaseEnv.reset`.

    By default, each time you call the "step" function a different mix is used. Mixes, by default
    are looped through always in the same order. You can see the Examples section for information
    about control of these


    Examples
    --------
    In this section we present some common use of the MultiMix environment.

    **Basic Usage**

    You can think of a MultiMixEnvironment as any :class:`Environment`. So this is a perfectly
    valid way to use a MultiMix:

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # define an agent like in any environment
        agent = RandomAgent(multimix_env.action_space)

        # and now you can do the open ai gym loop
        NB_EPISODE = 10
        for i in range(NB_EPISODE):
            obs = multimix_env.reset()
            # each time "reset" is called, another mix is used.
            reward = multimix_env.reward_range[0]
            done = False
            while not done:
                act = agent.act(obs, reward, done)
                obs, reward, done, info = multimix_env.step(act)

    **Use each mix one after the other**

    In case you want to study each mix independently, you can iterate through the MultiMix
    in a pythonic way. This makes it easy to perform, for example, 10 episode for a given mix
    before passing to the next one.

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        NB_EPISODE = 10
        for mix in multimix_env:
            # mix is a regular environment, you can do whatever you want with it
            # for example
            for i in range(NB_EPISODE):
                obs = multimix_env.reset()
                # each time "reset" is called, another mix is used.
                reward = multimix_env.reward_range[0]
                done = False
                while not done:
                    act = agent.act(obs, reward, done)
                    obs, reward, done, info = multimix_env.step(act)


    **Selecting a given Mix**

    Sometimes it might be interesting to study only a given mix.
    For that you can use the `[]` operator to select only a given mix (which is a grid2op environment)
    and use it as you would.

    This can be done with:

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # define an agent like in any environment
        agent = RandomAgent(multimix_env.action_space)

        # list all available mixes:
        mixes_names = list(multimix_env.keys())

        # and now supposes we want to study only the first one
        mix = multimix_env[mixes_names[0]]

        # and now you can do the open ai gym loop, or anything you want with it
        NB_EPISODE = 10
        for i in range(NB_EPISODE):
            obs = mix.reset()
            # each time "reset" is called, another mix is used.
            reward = mix.reward_range[0]
            done = False
            while not done:
                act = agent.act(obs, reward, done)
                obs, reward, done, info = mix.step(act)

    **Using the Runner**

    For MultiMixEnvironment using the :class:`grid2op.Runner.Runner` cannot be done in a
    straightforward manner. Here we give an example on how to do it.

    .. code-block:: python

        import os
        import grid2op
        from grid2op.Agent import RandomAgent

        # we use an example of a multimix dataset attached with grid2op pacakage
        multimix_env = grid2op.make("l2rpn_neurips_2020_track2", test=True)

        # you can use the runner as following
        PATH = "PATH/WHERE/YOU/WANT/TO/SAVE/THE/RESULTS"
        for mix in multimix_env:
            runner = Runner(**mix.get_params_for_runner(), agentClass=RandomAgent)
            runner.run(nb_episode=1,
                       path_save=os.path.join(PATH,mix.name))

    """
    def __init__(self,
                 envs_dir,
                 _add_to_name="",  # internal, for test only, do not use !
                 **kwargs):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self.current_env = None
        self.env_index = None
        self.mix_envs = []

        # Special case handling for backend 
        backendClass = None
        if "backend" in kwargs:
            backendClass = type(kwargs["backend"])
            del kwargs["backend"]

        # Inline import to prevent cyclical import
        from grid2op.MakeEnv.Make import make

        try:
            for env_dir in sorted(os.listdir(envs_dir)):
                env_path = os.path.join(envs_dir, env_dir)            
                if not os.path.isdir(env_path):
                    continue
                # Special case for backend
                if backendClass is not None:
                    env = make(env_path,
                               backend=backendClass(),
                               _add_to_name=_add_to_name,
                               **kwargs)
                else:
                    env = make(env_path, **kwargs)
                
                self.mix_envs.append(env)
        except Exception as e:
            err_msg = "MultiMix environment creation failed: {}".format(e)
            raise EnvError(err_msg)

        if len(self.mix_envs) == 0:
            err_msg = "MultiMix envs_dir did not contain any valid env"
            raise EnvError(err_msg)

        self.env_index = 0
        self.current_env = self.mix_envs[self.env_index]
        # Make sure GridObject class attributes are set from first env
        # Should be fine since the grid is the same for all envs
        multi_env_name = os.path.basename(os.path.abspath(envs_dir)) + _add_to_name
        save_env_name = self.current_env.env_name
        self.current_env.env_name = multi_env_name
        self.__class__ = self.init_grid(self.current_env)
        self.current_env.env_name = save_env_name

    @property
    def current_index(self):
        return self.env_index

    def __len__(self):
        return len(self.mix_envs)

    def __iter__(self):
        """
        Operator __iter__ overload to make a ``MultiMixEnvironment`` iterable

        .. code-block:: python

            import grid2op
            from grid2op.Environment import MultiMixEnvironment
            from grid2op.Runner import Runner

            mm_env = MultiMixEnvironment("/path/to/multi/dataset/folder")
            
            for env in mm_env:
                run_p = env.get_params_for_runner()
                runner = Runner(**run_p)
                runner.run(nb_episode=1, max_iter=-1)
        """
        self.env_index = 0
        return self

    def __next__(self):
        if self.env_index < len(self.mix_envs):
            r = self.mix_envs[self.env_index]
            self.env_index = self.env_index + 1
            return r
        else:
            self.env_index = 0
            raise StopIteration

    def __getattr__(self, name):
        return getattr(self.current_env, name)

    def keys(self):
        for mix in self.mix_envs:
            yield mix.name

    def values(self):
        for mix in self.mix_envs:
            yield mix

    def items(self):
        for mix in self.mix_envs:
            yield mix.name, mix

    def copy(self):
        mix_envs = self.mix_envs
        self.mix_envs = None
        current_env = self.current_env
        self.current_env = None

        cls = self.__class__
        res = cls.__new__(cls)
        for k in self.__dict__:
            if k == "mix_envs" or k == "current_env":
                # this is handled elswhere
                continue
            setattr(res, k, copy.deepcopy(getattr(self, k)))
        res.mix_envs = [mix.copy() for mix in mix_envs]
        res.current_env = res.mix_envs[res.env_index]

        self.mix_envs = mix_envs
        self.current_env = current_env
        return res

    def __getitem__(self, key):
        """
        Operator [] overload for accessing underlying mixes by name

        .. code-block:: python

            import grid2op
            from grid2op.Environment import MultiMixEnvironment

            mm_env = MultiMixEnvironment("/path/to/multi/dataset/folder")

            mix1_env.name = mm_env["mix_1"]
            assert mix1_env == "mix_1"
            mix2_env.name = mm_env["mix_2"]
            assert mix2_env == "mix_2"
        """
        # Search for key
        for mix in self.mix_envs:
            if mix.name == key:
                return mix

        # Not found by name
        raise KeyError
    
    def reset(self, random=False):
        if random:
            self.env_index = self.space_prng.randint(len(self.mix_envs))
        else:
            self.env_index = (self.env_index + 1) % len(self.mix_envs)

        self.current_env = self.mix_envs[self.env_index]
        self.current_env.reset()
        return self.get_obs()

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control
        and to ease reproducible experiments.

        Parameters
        ----------
        seed: ``int``
           The seed to set.

        Returns
        ---------
        seeds: ``list``
            The seed used to set the prng (pseudo random number generator)
            for all environments, and each environment ``tuple`` seeds

        """
        try:
            seed = np.array(seed).astype(dt_int)
        except Exception as e:
            raise Grid2OpException("Cannot to seed with the seed provided." \
                                   "Make sure it can be converted to a" \
                                   "numpy 32 bits integer.")

        s = super().seed(seed)
        seeds = [s]
        max_dt_int = np.iinfo(dt_int).max
        for env in self.mix_envs:
            env_seed = self.space_prng.randint(max_dt_int)
            env_seeds = env.seed(env_seed)
            seeds.append(env_seeds)
        return seeds

    def set_chunk_size(self, new_chunk_size):
        for mix in self.mix_envs:
            mix.set_chunk_size(new_chunk_size)

    def set_id(self, id_):
        for mix in self.mix_envs:
            mix.set_id(id_)

    def deactivate_forecast(self):
        for mix in self.mix_envs:
            mix.deactivate_forecast()

    def reactivate_forecast(self):
        for mix in self.mix_envs:
            mix.reactivate_forecast()

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.
        Will propagate to all underlying mixes
        """
        for mix in self.mix_envs:
            mix.set_thermal_limit(thermal_limit)

    def __enter__(self):
        """
        Support *with-statement* for the environment.

        """
        return self

    def __exit__(self, *args):
        """
        Support *with-statement* for the environment.

        """
        self.close()
        # propagate exception
        return False

    def close(self):
        for mix in self.mix_envs:
            mix.close()

    def attach_layout(self, grid_layout):
        for mix in self.mix_envs:
            mix.attach_layout(grid_layout)
