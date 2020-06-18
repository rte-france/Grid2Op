# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import numpy as np

from grid2op.dtypes import dt_int, dt_float
from grid2op.Space import GridObjects, RandomObject
from grid2op.Exceptions import EnvError 

class MultiMixEnvironment(GridObjects, RandomObject):
    """
    This class represent a single powergrid configuration,
    backed by multiple enviromnents parameters and chronics

    It implements most of the BaseEnv public interface:
    so it can be used as a more classic environment.

    """
    def __init__(self,
                 envs_dir,
                 **kwargs):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self.current_env = None
        self.env_index = None
        self._envs = []

        # Inline import to prevent cyclical import
        from grid2op.MakeEnv.Make import make

        try:
            for env_dir in sorted(os.listdir(envs_dir)):
                env_path = os.path.join(envs_dir, env_dir)            
                if not os.path.isdir(env_path):
                    continue
                env = make(env_path, **kwargs)
                self._envs.append(env)
        except Exception as e:
            err_msg = "MultiMix environment creation failed: {}".format(e)
            raise EnvError(err_msg)

        if len(self._envs) == 0:
            err_msg = "MultiMix envs_dir did not contain any valid env"
            raise EnvError(err_msg)

        self.env_index = 0
        self.current_env = self._envs[self.env_index]
        # Make sure GridObject class attributes are set from first env
        # Should be fine since the grid is the same for all envs
        self.__class__ = self.init_grid(self.current_env)

    @property
    def current_index(self):
        return self.env_index

    def __getattr__(self, name):
        return getattr(self.current_env, name)

    def reset(self, random=False):
        if random:
            self.env_index = self.space_prng.randint(len(self._envs))
        else:
            self.env_index = (self.env_index + 1) % len(self._envs)
        
        self.current_env = self._envs[self.env_index]
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
                                   "numpy 64 integer.")

        s = super().seed(seed)
        seeds = [s]
        max_dt_int = np.iinfo(dt_int).max
        for env in self._envs:
            env_seed = self.space_prng.randint(max_dt_int)
            env_seeds = env.seed(env_seed)
            seeds.append(env_seeds)
        return seeds

    def deactivate_forecast(self):
        for e in self._envs:
            e.deactivate_forecast()

    def reactivate_forecast(self):
        for e in self._envs:
            e.reactivate_forecast()

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.
        Will propagate to all underlying environments
        """
        for e in self._envs:
            e.set_thermal_limit(thermal_limit)

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
        for e in self._envs:
            e.close()

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
        for e in self._envs:
            e.attach_layout(grid_layout)
