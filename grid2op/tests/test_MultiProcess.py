# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest
from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Environment import BaseMultiProcessEnvironment
from grid2op.Environment import SingleEnvMultiProcess
from grid2op.Environment import MultiEnvMultiProcess
from grid2op.Observation import CompleteObservation
import pdb


class TestBaseMultiProcessEnvironment(unittest.TestCase):
    def test_creation_multienv(self):
        nb_env = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                envs = [env for _ in range(nb_env)]
                multi_envs = BaseMultiProcessEnvironment(envs)

                obss, rewards, dones, infos = multi_envs.step(
                    [env.action_space() for _ in range(multi_envs.nb_env)]
                )
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                obss = multi_envs.reset()
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                # test some actions will not throw errors
                multi_envs.set_ff(7 * 288)
                multi_envs.set_chunk_size(128)
                obss = multi_envs.reset()
                seeds = multi_envs.get_seeds()
                multi_envs.close()

    def test_seeding(self):
        nb_env = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                envs = [env for _ in range(nb_env)]
                env.seed(2)
                multi_envs1 = BaseMultiProcessEnvironment(envs)
                seeds_1 = multi_envs1.get_seeds()
                multi_envs1.close()
                multi_envs2 = BaseMultiProcessEnvironment(envs)
                seeds_2 = multi_envs2.get_seeds()
                multi_envs2.close()
                env.seed(2)
                multi_envs3 = BaseMultiProcessEnvironment(envs)
                seeds_3 = multi_envs3.get_seeds()
                multi_envs3.close()
                assert np.all(seeds_1 == seeds_3)
                assert np.any(seeds_1 != seeds_2)

    def test_simulate(self):
        env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env1 = grid2op.make(env_name, test=True, _add_to_name=type(self).__name__)
            env2 = grid2op.make(env_name, test=True, _add_to_name=type(self).__name__)

        multi_env = BaseMultiProcessEnvironment([env1, env2])
        obss = multi_env.reset()

        # simulate
        actions = [env1.action_space(), env2.action_space()]
        sim_obss, sim_rs, sim_ds, sim_is = multi_env.simulate(actions)
        multi_env.close()
        env1.close()
        env2.close()


class TestSingleEnvMultiProcess(unittest.TestCase):
    def test_creation_multienv(self):
        nb_env = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                multi_envs = SingleEnvMultiProcess(env=env, nb_env=nb_env)

                obss, rewards, dones, infos = multi_envs.step(
                    [env.action_space() for _ in range(multi_envs.nb_env)]
                )
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                obss = multi_envs.reset()
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                # test some actions will not throw errors
                multi_envs.set_ff(7 * 288)
                multi_envs.set_chunk_size(128)
                obss = multi_envs.reset()
                seeds = multi_envs.get_seeds()
                multi_envs.close()

    def test_seeding(self):
        nb_env = 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                env.seed(2)
                multi_envs1 = SingleEnvMultiProcess(env=env, nb_env=nb_env)
                seeds_1 = multi_envs1.get_seeds()
                multi_envs1.close()
                multi_envs2 = SingleEnvMultiProcess(env=env, nb_env=nb_env)
                seeds_2 = multi_envs2.get_seeds()
                multi_envs2.close()
                env.seed(2)
                multi_envs3 = SingleEnvMultiProcess(env=env, nb_env=nb_env)
                seeds_3 = multi_envs3.get_seeds()
                multi_envs3.close()
                assert np.all(seeds_1 == seeds_3)
                assert np.any(seeds_1 != seeds_2)


class TestMultiEnvMultiProcess(unittest.TestCase):
    def test_creation_multienv(self):
        nb_envs = [1, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                envs = [env for _ in range(len(nb_envs))]
                multi_envs = MultiEnvMultiProcess(envs, nb_envs)

                obss, rewards, dones, infos = multi_envs.step(
                    [env.action_space() for _ in range(multi_envs.nb_env)]
                )
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                obss = multi_envs.reset()
                for ob in obss:
                    assert isinstance(ob, CompleteObservation)

                # test some actions will not throw errors
                multi_envs.set_ff(7 * 288)
                multi_envs.set_chunk_size(128)
                obss = multi_envs.reset()
                seeds = multi_envs.get_seeds()
                multi_envs.close()

    def test_seeding(self):
        nb_envs = [1, 1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__) as env:
                envs = [env for _ in range(len(nb_envs))]
                env.seed(2)
                multi_envs1 = MultiEnvMultiProcess(envs, nb_envs)
                seeds_1 = multi_envs1.get_seeds()
                multi_envs1.close()
                multi_envs2 = MultiEnvMultiProcess(envs, nb_envs)
                seeds_2 = multi_envs2.get_seeds()
                multi_envs2.close()
                env.seed(2)
                multi_envs3 = MultiEnvMultiProcess(envs, nb_envs)
                seeds_3 = multi_envs3.get_seeds()
                multi_envs3.close()
                assert np.all(seeds_1 == seeds_3)
                assert np.any(seeds_1 != seeds_2)


if __name__ == "__main__":
    unittest.main()
