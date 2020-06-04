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
                 envs_dir):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self.current_env = None
        self._envs = []

        # Inline import to prevent cyclical import
        from grid2op.MakeEnv.Make import make

        try:
            for env_dir in sorted(os.listdir(envs_dir)):
                env_path = os.path.join(envs_dir, env_dir)            
                if not os.path.isdir(env_path):
                    continue
                env = make(env_path)
                self._envs.append(env)
        except Exception as e:
            err_msg = "MultiMix environment creation failed: {}".format(e)
            raise EnvError(err_msg)

        if len(self._envs) == 0:
            err_msg = "MultiMix envs_dir did not contain any valid env"
            raise EnvError(err_msg)
            
        self.current_env = self._envs[0]
        # Make sure GridObject class attributes are set from first env
        # Shouldbe fine since the grid is the same for all envs
        self.__class__ = self.init_grid(self.current_env)

    def __getattr__(self, name):
        return getattr(self.current_env, name)

    def reset(self):
        rnd_env_idx = np.random.randint(len(self._envs))
        self.current_env = self._envs[rnd_env_idx]
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
        seed: ``tuple``
            The seed used to set the prng (pseudo random number generator) 
            for all environments

        """
        try:
            seed = np.array(seed).astype(dt_int)
        except Exception as e:
            raise Grid2OpException("Cannot to seed with the seed provided." \
                                   "Make sure it can be converted to a" \
                                   "numpy 64 integer.")

        return super().seed(seed)

    def deactivate_forecast(self):
        self.current_env.deactive_forecast()

    def reactivate_forecast(self):
        self.current_env.reactivate_forecast()

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.
        Will propagate to all underlying environments
        """
        for e in self._envs:
            e.set_thermal_limit(thermal_limit)

    def get_obs(self):
        """
        Return the observations of the current environment made by 
        the :class:`grid2op.BaseAgent.BaseAgent`.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The current BaseObservation given to the 
            :class:`grid2op.BaseAgent.BaseAgent` / bot / controler.
        """
        return self.current_env.get_obs()

    def get_thermal_limit(self):
        """
        Get the current environment thermal limit in amps
        """
        return self.current_env.get_thermal_limit()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple:
           - (observation, reward, done, info).

        If the :class:`grid2op.BaseAction.BaseAction` is illegal or ambiguous,
        the step is performed, but the action is
        replaced with a "do nothing" action.

        Parameters
        ----------
            action: :class:`grid2op.Action.Action`
                an action provided by the agent that is applied 
                on the underlying through the backend.

        Returns
        -------
            observation: :class:`grid2op.Observation.Observation`
                agent's observation of the current environment

            reward: ``float``
                amount of reward returned after previous action

            done: ``bool``
                whether the episode has ended, 
                in which case further step() calls is undefined behavior

            info: ``dict``
                contains auxiliary diagnostic information

        """
        return self.current_env.step(action)

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

    def fast_forward_chronics(self, nb_timestep):
        """
        This method allows you to skip some time step at the 
        beginning of the chronics.

        This is usefull at the beginning of the training, 
        if you want your agent to learn on more diverse scenarios.
        Indeed, the data provided in the chronics usually starts 
        always at the same date time (for example Jan 1st at 00:00). 
        This can lead to suboptimal exploration, as during this phase, 
        only a few time steps are managed by the agent, 
        so in general these few time steps will correspond to grid 
        state around Jan 1st at 00:00.

        Parameters
        ----------
        nb_timestep: ``int``
            Number of time step to "fast forward"

        """
        self.current_env.fast_forward_chronics(nb_timestep)
