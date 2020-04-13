# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This module presents the class that can be modified to adapt (on the fly) the setpoint of the generators with
respect to the voltage magnitude.

Voltage magnitude plays a big part in real time operation process. Bad voltages can lead to different kind of problem
varying from:

- high losses (the higher the voltages, the lower the losses in general)
- equipment failures (typically if the voltages are too high)
- a really bad "quality of electricity" for consumers (if voltages is too low)
- partial or total blackout in case of voltage collapse (mainly if voltages are too low)

We wanted, in this package, to treat the voltages setpoint of the generators differently from the other
part of the game. This module exposes the main class to do this.
"""

from grid2op.VoltageControler.BaseVoltageController import BaseVoltageController


class ControlVoltageFromFile(BaseVoltageController):
    """
    This class is the most basic controler for the voltages. Basically, what it does is read the voltages from the
    chronics.

    If the voltages are not on the chronics (missing files), it will not change the voltage setpoints at all.
    """
    def __init__(self, gridobj, controler_backend):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Structure of the powergrid

        envbackend: :class:`grid2op.Backend.Backend`
            An instanciated backend to perform some computation on a powergrid, before taking some actions.

        """
        BaseVoltageController.__init__(self, gridobj=gridobj, controler_backend=controler_backend)

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
        # TODO add a "reward" and "done" for RL voltage controler
        if prod_v_chronics is not None:
            res = self.action_space({"injection": {"prod_v": prod_v_chronics}})
        else:
            res = self.action_space()
        return res
