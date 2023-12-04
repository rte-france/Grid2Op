# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op.Observation import CompleteObservation


class Issue235TesterObs(CompleteObservation):
    def __init__(self,
                 obs_env=None,
                 action_helper=None,
                 random_prng=None,
                 kwargs_env=None):
        CompleteObservation.__init__(
            self, obs_env, action_helper, random_prng=random_prng, kwargs_env=kwargs_env
        )
        self._is_updated = False

    def update(self, env, with_forecast=True):
        self._is_updated = True
        super().update(env, with_forecast)


class Issue235Tester(unittest.TestCase):
    """
    This bug was due to the environment that updated the observation even when it diverges.

    In this test i checked that the `update` method of the observation is not called even when I simulate
    an action that lead to divergence of the powerflow.
    """

    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_nm = "l2rpn_icaps_2021"
            # from lightsim2grid import LightSimBackend
            # backend=LightSimBackend(),
            self.env = grid2op.make(
                env_nm, test=True, observation_class=Issue235TesterObs,
                _add_to_name=type(self).__name__
            )

        # now set the right observation class for the simulate action
        hack_obs_cls = Issue235TesterObs.init_grid(type(self.env))
        self.env._observation_space.obs_env.current_obs = hack_obs_cls()
        self.env._observation_space.obs_env.current_obs_init = hack_obs_cls()
        # now do regular gri2op stuff
        self.env.seed(0)
        self.env.reset()

    def test_diverging_action(self):
        final_dict = {
            "generators_id": [(19, 1)],
            "loads_id": [(30, 2)],
            "lines_or_id": [(58, 1)],
            "lines_ex_id": [(46, 1), (47, 2)],
        }
        action = self.env.action_space({"set_bus": final_dict})
        obs = self.env.reset()
        simobs, simr, simd, siminfo = obs.simulate(action, time_step=0)
        assert simd
        assert np.all(simobs.gen_p == 0.0)
        assert not simobs._is_updated
