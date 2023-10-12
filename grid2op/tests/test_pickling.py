# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import copy

import multiprocessing as mp

import grid2op
from grid2op.gym_compat import (
    ContinuousToDiscreteConverter,
    GymEnv,
    MultiToTupleConverter,
    ScalerAttrConverter,
)


with warnings.catch_warnings():
    # this needs to be imported in the main module for multiprocessing to work "approximately"
    warnings.filterwarnings("ignore")
    _ = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=__name__+"for_mp_test")


class TestMultiProc(unittest.TestCase):
    @staticmethod
    def f(env_gym):
        return env_gym.action_space.sample()

    @staticmethod
    def g(env_gym):
        act = env_gym.action_space.sample()
        return env_gym.step(act)[0]

    def test_basic(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "l2rpn_case14_sandbox", test=True, _add_to_name=__name__+"for_mp_test"
            )
        env_gym = GymEnv(env)

        obs_gym, *_ = env_gym.reset()

        # 3. (optional) customize it (see section above for more information)
        ## customize action space
        env_gym.action_space = env_gym.action_space.ignore_attr("set_bus").ignore_attr(
            "set_line_status"
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", ContinuousToDiscreteConverter(nb_bins=11)
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "change_bus", MultiToTupleConverter()
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "change_line_status", MultiToTupleConverter()
        )
        env_gym.action_space = env_gym.action_space.reencode_space(
            "redispatch", MultiToTupleConverter()
        )

        ## customize observation space
        ob_space = env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"]
        )
        ob_space = ob_space.reencode_space(
            "actual_dispatch", ScalerAttrConverter(substract=0.0, divide=env.gen_pmax)
        )
        ob_space = ob_space.reencode_space(
            "gen_p", ScalerAttrConverter(substract=0.0, divide=env.gen_pmax)
        )
        ob_space = ob_space.reencode_space(
            "load_p",
            ScalerAttrConverter(
                substract=obs_gym["load_p"], divide=0.5 * obs_gym["load_p"]
            ),
        )
        env_gym.observation_space = ob_space

        ctx = mp.get_context("spawn")
        env_gym1 = copy.deepcopy(env_gym)
        env_gym2 = copy.deepcopy(env_gym)
        with ctx.Pool(2) as p:
            p.map(TestMultiProc.f, [env_gym1, env_gym2])

        with ctx.Pool(2) as p:
            p.map(TestMultiProc.g, [env_gym1, env_gym2])


if __name__ == "__main__":
    unittest.main()
