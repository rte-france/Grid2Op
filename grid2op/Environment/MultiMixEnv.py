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
    backed by multiple enviromnents parameters and chronics

    It implements most of the BaseEnv public interface:
    so it can be used as a more classic environment.

    # TODO example on how to use it

    """
    def __init__(self,
                 envs_dir,
                 _for_copy=False,
                 **kwargs):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self.current_env = None
        self.env_index = None
        self.mix_envs = []

        if _for_copy:
            # used for making copy of this environment
            # do not set
            return

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
        multi_env_name = os.path.basename(os.path.abspath(envs_dir))
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
        Will propagate to all underlying environments
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
        """
        Compare to the method of the base class, this one performs a check.
        This method must be called after initialization.

        Parameters
        ----------
        grid_layout

        Returns
        -------

        """
        for mix in self.mix_envs:
            mix.attach_layout(grid_layout)
