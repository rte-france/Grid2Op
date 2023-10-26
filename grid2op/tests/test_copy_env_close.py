# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import gc
import sys
import warnings
import unittest

# see https://code.activestate.com/recipes/577504/
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

import grid2op
from grid2op.Exceptions import EnvError
from grid2op.Backend import Backend
from pandapower.auxiliary import pandapowerNet


class TestDanglingRef(unittest.TestCase):
    def _clean_envs(self):
        for obj_ in gc.get_objects():
            if (
                isinstance(obj_, grid2op.Environment.BaseEnv)
                or isinstance(obj_, grid2op.Environment.BaseMultiProcessEnvironment)
                or isinstance(obj_, grid2op.Environment.MultiMixEnvironment)
                or isinstance(obj_, grid2op.Observation.BaseObservation)
            ):
                del obj_

    def setUp(self) -> None:
        # make sure that there is no "dangling" reference to any environment
        current_len = len(gc.get_objects()) + 1
        while len(gc.get_objects()) != current_len:
            self._clean_envs()
            current_len = len(gc.get_objects())
            gc.collect()
        gc.collect()

    def test_dangling_reference(self):
        nb_env_init = len(
            [o for o in gc.get_objects() if isinstance(o, grid2op.Environment.BaseEnv)]
        )
        nb_backend_init = len([o for o in gc.get_objects() if isinstance(o, Backend)])
        nb_ppnet_init = len(
            [o for o in gc.get_objects() if isinstance(o, pandapowerNet)]
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        nb_env_before = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_before = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        nb_ppnet_before = (
            len([o for o in gc.get_objects() if isinstance(o, pandapowerNet)])
            - nb_ppnet_init
        )

        assert (
            nb_env_before == 2
        ), f"there should be 2 environments, but we found {nb_env_before}"
        assert (
            nb_backend_before == 2
        ), f"there should be 2 backends, but we found {nb_backend_before}"
        assert (
            nb_ppnet_before == 4
        ), f"there should be 4 pp networks, but we found {nb_ppnet_before}"
        # there are 4 pp nets because PandaPowerBackend keeps a copy of the initial grid for faster reset
        # and it's copied for both the env backend and the obs_env backend

        # make a copy
        env_cpy = env.copy()
        nb_env_after = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_after = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        nb_ppnet_after = (
            len([o for o in gc.get_objects() if isinstance(o, pandapowerNet)])
            - nb_ppnet_init
        )
        assert (
            nb_env_after == 4
        ), f"there should be 4 environments after copy, but we found {nb_env_after}"
        assert (
            nb_backend_after == 4
        ), f"there should be 4 backend after copy, but we found {nb_backend_after}"
        assert (
            nb_ppnet_after == 8
        ), f"there should be 8 pp networks after copy, but we found {nb_ppnet_after}"

        # reset the copied environment
        obs_cpy = env_cpy.reset()
        nb_env_after_reset = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_after_reset = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        assert (
            nb_env_after_reset == 4
        ), f"there should be 4 environments after reset, but we found {nb_env_after_reset}"
        assert (
            nb_backend_after_reset == 4
        ), f"there should be 4 backends after reset, but we found {nb_backend_after_reset}"

        # call step (on the copied env)
        obs_cpy, reward, done, info = env_cpy.step(env_cpy.action_space())
        nb_env_after_step = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_after_step = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        assert (
            nb_env_after_step == 4
        ), f"there should be 4 environments after step, but we found {nb_env_after_step}"
        assert (
            nb_backend_after_step == 4
        ), f"there should be 4 backends after step, but we found {nb_backend_after_step}"

        # call steps on init env
        obs, reward, done, info = env.step(env_cpy.action_space())
        nb_env_after_step = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_after_step = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        assert (
            nb_env_after_step == 4
        ), f"there should be 4 environments after step, but we found {nb_env_after_step}"
        assert (
            nb_backend_after_step == 4
        ), f"there should be 4 backends after step, but we found {nb_backend_after_step}"

        # now i close the initial environment, and check that everything is working as expected
        env.close()
        del env
        gc.collect()

        nb_env_after_close = (
            len(
                [
                    o
                    for o in gc.get_objects()
                    if isinstance(o, grid2op.Environment.BaseEnv)
                ]
            )
            - nb_env_init
        )
        nb_backend_after_close = (
            len([o for o in gc.get_objects() if isinstance(o, Backend)])
            - nb_backend_init
        )
        nb_ppnet_after_close = (
            len([o for o in gc.get_objects() if isinstance(o, pandapowerNet)])
            - nb_ppnet_init
        )

        assert (
            nb_env_after_close == 3
        ), f"there should be 3 environments after close, but we found {nb_env_after_close}"
        # the "obs_env" of the observation cannot be collected, as it's used on the observation...
        assert (
            nb_backend_after_close == 2
        ), f"there should be 2 backends after close, but we found {nb_backend_after_close}"
        assert (
            nb_ppnet_after_close == 4
        ), f"there should be 4 pp networks after close, but we found {nb_ppnet_after_close}"
        # but the "grid" of the "obs_env" is definitely cleaned up

        # now check i can properly do step, reset and simulate
        obs_cpy, reward, done, info = env_cpy.step(env_cpy.action_space())
        _ = obs_cpy.simulate(env_cpy.action_space())
        obs_cpy = env_cpy.reset()

        # finally I checked that I cannot use simulate on the closed environment
        with self.assertRaises(EnvError):
            _ = obs.simulate(env_cpy.action_space())


if __name__ == "__main__":
    unittest.main()
