# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import numpy as np

#!/usr/bin/env python3

import grid2op
import unittest
import numpy as np
import warnings
from grid2op.Parameters import Parameters
import pdb


class Issue140Tester(unittest.TestCase):
    def test_issue_140(self):

        # TODO change that
        envs = grid2op.list_available_local_env()
        env_name = "l2rpn_neurips_2020_track1_small"
        if env_name not in envs:
            # the environment is not downloaded, I skip this test
            self.skipTest("{} is not downloaded".format(env_name))

        param = Parameters()
        # test was originally designed with these values
        param.init_from_dict(
            {
                "NO_OVERFLOW_DISCONNECTION": False,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 3,
                "NB_TIMESTEP_COOLDOWN_SUB": 3,
                "NB_TIMESTEP_COOLDOWN_LINE": 3,
                "HARD_OVERFLOW_THRESHOLD": 200.0,
                "NB_TIMESTEP_RECONNECTION": 12,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                env_name,
                param=param,
                _add_to_name=type(self).__name__,
            )
        ts_per_chronics = 2016

        seed = 725
        np.random.seed(seed)
        env.seed(seed)

        env.set_id(np.random.randint(100000))
        _ = env.reset()

        max_day = (env.chronics_handler.max_timestep() - 2016) // 288
        start_timestep = np.random.randint(max_day) * 288 - 1  # start at 00:00
        if start_timestep > 0:
            env.fast_forward_chronics(start_timestep)

        obs = env.get_obs()

        done = False
        steps = 0
        do_nothing = env.action_space({})

        for i in range(119):
            obs, reward, done, info = env.step(do_nothing)
            assert not done

        act0 = env.action_space(
            {
                "set_bus": {
                    "lines_ex_id": [(41, 2), (42, 1), (57, 1)],
                    "lines_or_id": [(44, 2)],
                    "generators_id": [(16, 2)],
                }
            }
        )
        act1 = env.action_space(
            {
                "set_bus": {
                    "lines_ex_id": [
                        (17, 2),
                        (18, 1),
                        (19, 1),
                        (20, 2),
                        (21, 2),
                    ],
                    "lines_or_id": [
                        (22, 2),
                        (23, 2),
                        (27, 1),
                        (28, 1),
                        (48, 2),
                        (49, 2),
                        (54, 1),
                    ],
                    "generators_id": [(5, 2), (6, 1), (7, 1), (8, 1)],
                    "loads_id": [(17, 2)],
                }
            }
        )
        act2 = env.action_space(
            {
                "set_bus": {
                    "lines_ex_id": [(36, 2), (37, 2), (38, 2), (39, 2)],
                    "lines_or_id": [(40, 1), (41, 1)],
                    "generators_id": [(14, 1)],
                    "loads_id": [(27, 1)],
                }
            }
        )
        obs, reward, done, info = env.step(act0)
        obs, reward, done, info = env.step(act1)
        assert not done

        simulate_obs0, simulate_reward0, simulate_done0, simulate_info0 = obs.simulate(
            do_nothing
        )
        simulate_obs1, simulate_reward1, simulate_done1, simulate_info1 = obs.simulate(
            act2
        )
        obs, reward, done, info = env.step(act2)

        assert simulate_done0 is False
        assert simulate_done1 is True
        assert done is True

        assert info["is_illegal"] == False
        assert simulate_info1["is_illegal"] == False


if __name__ == "__main__":
    unittest.main()
