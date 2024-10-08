# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import pandapower as pp
import tempfile
import os
from pathlib import Path
import warnings
import copy
import numpy as np

from helper_path_test import PATH_DATA_TEST
import grid2op
from grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from grid2op.Action.playableAction import PlayableAction
from grid2op.Observation.completeObservation import CompleteObservation
from grid2op.Reward.flatReward import FlatReward
from grid2op.Rules.DefaultRules import DefaultRules
from grid2op.Chronics.multiFolder import Multifolder
from grid2op.Chronics.gridStateFromFileWithForecasts import GridStateFromFileWithForecasts
from grid2op.Chronics import ChangeNothing


class Issue617Tester(unittest.TestCase):
    def setUp(self):
        self.env_name = "l2rpn_case14_sandbox"
        # create first env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
        root_path = Path(os.path.abspath(PATH_DATA_TEST))
        self.env_path = tempfile.TemporaryDirectory(dir=root_path)
        self.tol = 1e-6
        
    def tearDown(self) -> None:
        self.env_path.cleanup()
        return super().tearDown()

    def create_config(self, env_path:Path, network, **kwargs):
        thermal_limits = [10_000. * el for el in network.line.max_i_ka] # Thermal Limit in Amps (A)
        with open(Path(env_path.name) / "config.py", "w") as config:
            # Import Statements
            config.writelines(
                [f"from {value.__module__} import {value.__name__}\n" for value in kwargs.values() if hasattr(value, "__module__")]
            )

            # Config Dictionary
            config.writelines(
                ["config = {\n"] +
                [f"'{k}':{getattr(v,'__name__', 'None')},\n" for k,v in kwargs.items()] +
                [f"'thermal_limits':{thermal_limits}\n"] + 
                ["}\n"]
            )
        return thermal_limits

    def create_pp_net(self):
        network = pp.create_empty_network()
        pp.create_buses(network, nr_buses=2, vn_kv=20.0)
        pp.create_gen(network, bus=0, p_mw=10.0, min_p_mw=-1e9, max_p_mw=1e9, slack=True, slack_weight=1.0)
        pp.create_line(network, from_bus=0, to_bus=1, length_km=10.0, std_type="NAYY 4x50 SE")
        pp.create_load(network, bus=1, p_mw=10.0, controllable=False)
        pp.to_json(network, Path(self.env_path.name) / "grid.json")
        return network
    
    def test_can_make_env(self):
        network = self.create_pp_net()
        thermal_limits = self.create_config(self.env_path,
                  network,
                  backend=PandaPowerBackend,
                  action=PlayableAction,
                  observation_class=CompleteObservation,
                  reward_class=FlatReward,
                  gamerules_class=DefaultRules,
                  chronics_class=Multifolder,
                  grid_value_class=GridStateFromFileWithForecasts,
                  voltagecontroler_class=None,
                  names_chronics_to_grid=None)

        pp.runpp(network, numba=True, lightsim2grid=False, max_iteration=10, distributed_slack=False, init="dc", check_connectivity=False)
        assert network.converged
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.env_path.name, chronics_class=ChangeNothing)
        assert (np.abs(env.get_thermal_limit() - thermal_limits) <= 1e-6).all()
        obs = env.reset()
        assert (np.abs(obs.p_or - network.res_line["p_from_mw"]) <= self.tol).all()
        assert (np.abs(obs.q_or - network.res_line["q_from_mvar"]) <= self.tol).all()
        assert (np.abs(obs.a_or - 1000. * network.res_line["i_from_ka"]) <= self.tol).all()
        obs, reward, done, info = env.step(env.action_space())
        assert (np.abs(obs.p_or - network.res_line["p_from_mw"]) <= self.tol).all()
        assert (np.abs(obs.q_or - network.res_line["q_from_mvar"]) <= self.tol).all()
        assert (np.abs(obs.a_or - 1000. * network.res_line["i_from_ka"]) <= self.tol).all()

    
if __name__ == "__main__":
    unittest.main()
