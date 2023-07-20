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
from grid2op.Runner import Runner
from grid2op.Observation import (CompleteObservation, NoisyObservation)


class TestNoisy(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage", test=True, observation_class=NoisyObservation
            )
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

    def test_with_copy(self):
        env_cpy = self.env.copy()

        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs, *_ = self.env.step(self.env.action_space())
        obs = self.env.reset()
        obs_cpy = env_cpy.reset()
        self._obs_equals(obs_cpy, obs)

        obs, *_ = self.env.step(self.env.action_space())
        obs_cpy, *_ = env_cpy.step(self.env.action_space())
        self._obs_equals(obs_cpy, obs)

    def test_simulate(self):
        sim_o, *_ = self.obs.simulate(self.env.action_space())
        assert type(sim_o).env_name == "educ_case14_storage"
        assert isinstance(sim_o, CompleteObservation)

        # test that it is reproducible
        self.env.seed(0)
        self.env.set_id(0)
        as_ref = self.env.reset()  # should match self.obs
        sim_o2, *_ = as_ref.simulate(self.env.action_space())
        self._obs_equals(sim_o2, sim_o)

        # test that it is the same as non stochastic observation
        # (simulate is based on forecast, not on actual environment state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "educ_case14_storage", test=True, observation_class=CompleteObservation
            )
        env.seed(0)
        env.set_id(0)
        obs = env.reset()
        sim_o3, *_ = obs.simulate(self.env.action_space())
        self._obs_equals(sim_o3, sim_o)

    def test_runner(self):
        runner = Runner(**self.env.get_params_for_runner())
        # check it's the same when seed is the same
        res = runner.run(
            nb_episode=1,
            max_iter=10,
            episode_id=[0],
            env_seeds=[0],
            add_detailed_output=True,
        )
        res2 = runner.run(
            nb_episode=1,
            max_iter=10,
            episode_id=[0],
            env_seeds=[0],
            add_detailed_output=True,
        )

        self._obs_equals(res[0][-1].observations[0], self.obs)
        for el in range(10):
            obs1 = res[0][-1].observations[el]
            obs2 = res2[0][-1].observations[el]
            self._obs_equals(obs1, obs2)

        # check it's different when seed is different
        res3 = runner.run(
            nb_episode=1,
            max_iter=10,
            episode_id=[0],
            env_seeds=[1],
            add_detailed_output=True,
        )
        for el in range(10):
            obs1 = res[0][-1].observations[el]
            obs3 = res3[0][-1].observations[el]
            with self.assertRaises(AssertionError):
                self._obs_equals(obs1, obs3)


class TestNoisyDiffParams(TestNoisy):
    def setUp(self) -> None:
        kwargs_observation = {"sigma_load_p": 1.0, "sigma_gen_p": 0.1}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                observation_class=NoisyObservation,
                kwargs_observation=kwargs_observation,
            )
        self.env.seed(0)
        self.env.set_id(0)
        self.obs = self.env.reset()

    def test_param_working(self):
        # change the kwargs to make sure it has an impact
        kwargs_observation = {"sigma_load_p": 0.1, "sigma_gen_p": 1.0}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "educ_case14_storage",
                test=True,
                observation_class=NoisyObservation,
                kwargs_observation=kwargs_observation,
            )
        env.seed(0)
        env.set_id(0)
        obs = env.reset()
        with self.assertRaises(AssertionError):
            self._obs_equals(obs, self.obs)


# TODO next: have a powerflow there to compute the outcome of the state
# after the modification

if __name__ == "__main__":
    unittest.main()
