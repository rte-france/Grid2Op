# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from abc import ABC, abstractmethod
import numpy as np
import copy

from grid2op.dtypes import dt_int
from grid2op.Action import VoltageOnlyAction, ActionSpace
from grid2op.Rules import AlwaysLegal
from grid2op.Space import RandomObject


class BaseVoltageController(RandomObject, ABC):
    """
    This class is the most basic controler for the voltages. Basically, what it does is read the voltages from the
    chronics.

    If the voltages are not on the chronics (missing files), it will not change the voltage setpoints at all.
    """

    def __init__(self, gridobj, controler_backend, actionSpace_cls, _local_dir_cls=None):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.Gridobject`
            Structure of the powergrid

        controler_backend: :class:`grid2op.Backend.Backend`
            An instanciated backend to perform some computation on a powergrid, before taking some actions.

        """
        RandomObject.__init__(self)
        legal_act = AlwaysLegal()
        self._actionSpace_cls = actionSpace_cls
        self.action_space = actionSpace_cls(
            gridobj=gridobj,
            actionClass=VoltageOnlyAction,
            legal_action=legal_act,
            _local_dir_cls=_local_dir_cls
        )

    def _custom_deepcopy_for_copy(self, new_obj):
        RandomObject._custom_deepcopy_for_copy(self, new_obj)
        new_obj._actionSpace_cls = self._actionSpace_cls
        legal_act = AlwaysLegal()
        new_obj.action_space = new_obj._actionSpace_cls(
            gridobj=self._actionSpace_cls,
            actionClass=VoltageOnlyAction,
            legal_action=legal_act,
        )

    def copy(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Make a (deep) copy of this instance.
        """
        res = type(self).__new__(type(self))
        self._custom_deepcopy_for_copy(res)
        return res

    def attach_layout(self, grid_layout):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        """
        self.action_space.attach_layout(grid_layout)

    def seed(self, seed):
        """
        Used to seed the voltage controler class

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        """
        me_seed = super().seed(seed)
        max_int = np.iinfo(dt_int).max
        seed_space = self.space_prng.randint(max_int)
        space_seed = self.action_space.seed(seed_space)
        return me_seed, space_seed

    @abstractmethod
    def fix_voltage(self, observation, agent_action, env_action, prod_v_chronics):
        """
        This method must be overloaded to change the behaviour of the generator setpoint for time t+1.

        This simple class will:

        - do nothing if the vector `prod_v_chronics` is None
        - set the generator setpoint to the value in prod_v_chronics

        Basically, this class is pretty fast, but does nothing interesting, beside looking at the data in files.

        More general class can use, to adapt the voltage setpoint:

        - `observation` the observation (receive by the agent) at time t
        - `agent_action` the action of the agent at time t
        - `env_action` the modification of the environment at time t, that will be observed by the agent at time
          t+1
        - `prod_v_chronics` the new setpoint of the generators present in the data (if any, this can be None)

        To help this class, a :class:`grid2op.Backend.Backend` is available and can be used to perform simulation of
        potential impact of voltages setpoints.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took

        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.

        prod_v_chronics: ``numpy.ndarray``
            The next voltage setpoint present in the data (if any) or ``None`` if not.

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The new setpoint, in this case depending only on the prod_v_chronics.

        """
        pass

    def close(self):
        """If you require some "backend" to control the voltages, then you need to implement this
        (and also some specific things for the copy) to have it working correctly
        """
        pass
