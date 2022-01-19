# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import unittest
import warnings
import pdb

import grid2op
from grid2op.Observation.noisyObservation import NoisyObservation

class TestNoisy(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    observation_class=NoisyObservation)
        self.env.seed(0)
        self.env.set_id(0)
        self.obs = self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_create_ok(self):
        # simply test the creation
        pass

    def _obs_equals(self, obs1, obs2):
        assert np.all(obs1.load_p == obs2.load_p)
        assert np.all(obs1.load_q == obs2.load_q)
        assert np.all(obs1.gen_p == obs2.prod_p)
        assert np.all(obs1.gen_q == obs2.prod_q)
        assert np.all(obs1.a_or == obs2.a_or)
        assert np.all(obs1.a_ex == obs2.a_ex)
        assert np.all(obs1.p_or == obs2.p_or)
        assert np.all(obs1.p_ex == obs2.p_ex)
        assert np.all(obs1.q_or == obs2.q_or)
        assert np.all(obs1.q_ex == obs2.q_ex)
        assert np.all(obs1.storage_power == obs2.storage_power)

    def test_getobs_sameres(self):
        # simply test the creation
        obs0 = self.env.get_obs()
        obs1 = self.env.get_obs()
        assert np.all(obs0.load_p == obs1.load_p)
        assert np.any(obs0.load_p != self.env.backend.load_p)
        assert np.all(obs0.load_q == obs1.load_q)
        assert np.any(obs0.load_q != self.env.backend.load_q)
        assert np.all(obs0.gen_p == obs1.prod_p)
        assert np.any(obs0.gen_p != self.env.backend.prod_p)
        assert np.all(obs0.gen_q == obs1.prod_q)
        assert np.any(obs0.gen_q != self.env.backend.prod_q)
        assert np.all(obs0.a_or == obs1.a_or)
        assert np.any(obs0.a_or != self.env.backend.a_or)
        assert np.all(obs0.a_ex == obs1.a_ex)
        assert np.any(obs0.a_ex != self.env.backend.a_ex)
        assert np.all(obs0.p_ex == obs1.p_ex)
        assert np.any(obs0.p_ex != self.env.backend.p_ex)
        assert np.all(obs0.q_ex == obs1.q_ex)
        assert np.any(obs0.q_ex != self.env.backend.q_ex)
        assert np.all(obs0.p_or == obs1.p_or)
        assert np.any(obs0.p_or != self.env.backend.p_or)
        assert np.all(obs0.q_or == obs1.q_or)
        assert np.any(obs0.q_or != self.env.backend.q_or)

        assert np.all(obs0.storage_power == obs1.storage_power)
        assert np.any(obs0.storage_power != self.env._storage_power)

    def test_seed_works(self):
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        self._obs_equals(obs, self.obs)

    def test_seed_independant_previous(self):
        """test that the seed of a given episode is independant on what happened in the previous"""
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs = self.env.reset()

        self.env.seed(0)
        self.env.set_id(0)
        as_ref = self.env.reset()  # should match self.obs
        self._obs_equals(as_ref, self.obs)
        # don't do anything (instead of 3 steps)
        as_obs = self.env.reset()
        self._obs_equals(obs, as_obs)  # should match the case where I did 3 steps

# TODO test with copy (reproducible res)
# TODO: find way to change the std of underlying distributions
# TODO test that the forecast obs is always of type "CompleteObservation"

# TODO next: have a powerflow there to compute the outcome of the state 
# after the modification