# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import copy
from grid2op.Parameters import Parameters
from grid2op.Chronics import FromNPY
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
import numpy as np
import pdb

from grid2op.tests.helper_path_test import *


class TestNPYChronics(unittest.TestCase):
    """
    This class tests the possibility in grid2op to limit the number of call to "obs.simulate"
    """

    def setUp(self):
        self.env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env_ref = grid2op.make(self.env_name, test=True, _add_to_name=type(self).__name__)

        self.load_p = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p
        self.load_q = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q
        self.prod_p = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p
        self.prod_v = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v

    def tearDown(self) -> None:
        self.env_ref.close()

    def test_proper_start_end(self):
        """test i can create an environment with the FromNPY class"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 18,  # excluded
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )

        for ts in range(10):
            obs_ref, *_ = self.env_ref.step(env.action_space())
            assert np.all(
                obs_ref.gen_p[:-1] == self.prod_p[1 + ts, :-1]
            ), f"error at iteration {ts}"
            obs, *_ = env.step(env.action_space())
            assert np.all(obs_ref.gen_p == obs.gen_p), f"error at iteration {ts}"

        # test the "end"
        for ts in range(7):
            obs, *_ = env.step(env.action_space())
        obs, reward, done, info = env.step(env.action_space())
        assert done
        assert obs.max_step == 18
        with self.assertRaises(Grid2OpException):
            env.step(env.action_space())  # raises a Grid2OpException
        env.close()

    def test_proper_start_end_2(self):
        """test i can do as if the start was "later" """
        LAG = 5
        END = 18
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": LAG,
                    "i_end": END,
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )

        for ts in range(LAG):
            obs_ref, *_ = self.env_ref.step(env.action_space())

        for ts in range(END - LAG):
            obs_ref, *_ = self.env_ref.step(env.action_space())
            assert np.all(
                obs_ref.gen_p[:-1] == self.prod_p[1 + ts + LAG, :-1]
            ), f"error at iteration {ts}"
            obs, *_ = env.step(env.action_space())
            assert np.all(obs_ref.gen_p == obs.gen_p), f"error at iteration {ts}"
        assert obs.max_step == END
        with self.assertRaises(Grid2OpException):
            env.step(
                env.action_space()
            )  # raises a Grid2OpException because the env is done
        env.close()

    def test_iend_bigger_dim(self):
        max_step = 5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": self.load_p[:max_step, :],
                    "load_q": self.load_q[:max_step, :],
                    "prod_p": self.prod_p[:max_step, :],
                    "prod_v": self.prod_v[:max_step, :],
                },
            )
        assert env.chronics_handler.real_data._load_p.shape[0] == max_step
        for ts in range(
            max_step - 1
        ):  # -1 because one ts is "burnt" for the initialization
            obs, reward, done, info = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"

        obs, reward, done, info = env.step(env.action_space())
        assert done
        assert obs.max_step == max_step
        with self.assertRaises(Grid2OpException):
            env.step(
                env.action_space()
            )  # raises a Grid2OpException because the env is done
        env.close()

    def test_change_iend(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )
        assert env.chronics_handler.real_data._i_end == self.load_p.shape[0]
        env.chronics_handler.real_data.change_i_end(15)
        env.reset()
        assert env.chronics_handler.real_data._i_end == 15
        env.reset()
        assert env.chronics_handler.real_data._i_end == 15
        env.chronics_handler.real_data.change_i_end(25)
        assert env.chronics_handler.real_data._i_end == 15
        env.reset()
        assert env.chronics_handler.real_data._i_end == 25
        env.chronics_handler.real_data.change_i_end(None)  # reset default value
        env.reset()
        assert env.chronics_handler.real_data._i_end == self.load_p.shape[0]

        # now make sure it recomputes the maximum even if i change the size of the input arrays
        env.chronics_handler.real_data.change_chronics(
            self.load_p[:10], self.load_q[:10], self.prod_p[:10], self.prod_v[:10]
        )
        env.reset()
        assert env.chronics_handler.real_data._i_end == 10
        env.chronics_handler.real_data.change_chronics(
            self.load_p, self.load_q, self.prod_p, self.prod_v
        )
        env.reset()
        assert env.chronics_handler.real_data._i_end == self.load_p.shape[0]
        env.close()

    def test_change_istart(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )
        assert env.chronics_handler.real_data._i_start == 0
        env.chronics_handler.real_data.change_i_start(5)
        env.reset()
        assert env.chronics_handler.real_data._i_start == 5
        env.reset()
        assert env.chronics_handler.real_data._i_start == 5
        env.chronics_handler.real_data.change_i_start(10)
        assert env.chronics_handler.real_data._i_start == 5
        env.reset()
        assert env.chronics_handler.real_data._i_start == 10
        env.chronics_handler.real_data.change_i_start(None)  # reset default value
        env.reset()
        assert env.chronics_handler.real_data._i_start == 0
        env.close()

    def test_runner(self):
        max_step = 10
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": self.load_p[:max_step, :],
                    "load_q": self.load_q[:max_step, :],
                    "prod_p": self.prod_p[:max_step, :],
                    "prod_v": self.prod_v[:max_step, :],
                },
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore"
            )  # silence the UserWarning: Class FromNPY doesn't handle different input folder. "tell_id" method has no impact.
            # warnings.warn("Class {} doesn't handle different input folder. \"tell_id\" method has no impact."
            runner = Runner(**env.get_params_for_runner())
            res = runner.run(nb_episode=1)
            assert res[0][3] == 10  # number of time step selected
        env.close()

    def test_change_chronics(self):
        """test i can change the chronics"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 18,  # excluded
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )
        self.env_ref.reset()

        load_p = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p
        load_q = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q
        prod_p = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p
        prod_v = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v

        env.chronics_handler.real_data.change_chronics(load_p, load_q, prod_p, prod_v)
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
        env.reset()
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
        env.close()

    def test_with_env_copy(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                },
            )
        env_cpy = env.copy()
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
        for ts in range(10):
            obs_cpy, *_ = env_cpy.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs_cpy.gen_p[:-1]
            ), f"error at iteration {ts}"

        self.env_ref.reset()

        load_p = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p
        load_q = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q
        prod_p = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p
        prod_v = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v
        env.chronics_handler.real_data.change_chronics(load_p, load_q, prod_p, prod_v)
        env.reset()
        env_cpy.reset()
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
        for ts in range(10):
            obs_cpy, *_ = env_cpy.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs_cpy.gen_p[:-1]
            ), f"error at iteration {ts}"
        env.close()

    def test_forecast(self):
        load_p_f = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p_forecast
        load_q_f = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q_forecast
        prod_p_f = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p_forecast
        prod_v_f = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v_forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                    "load_p_forecast": load_p_f,
                    "load_q_forecast": load_q_f,
                    "prod_p_forecast": prod_p_f,
                    "prod_v_forecast": prod_v_f,
                },
            )

        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
            sim_obs, *_ = obs.simulate(env.action_space())
            assert np.all(
                prod_p_f[1 + ts, :-1] == sim_obs.gen_p[:-1]
            ), f"error at iteration {ts}"
            assert (
                sim_obs.minute_of_hour == (obs.minute_of_hour + 5) % 60
            ), f"error at iteration {ts}"
        env.close()

    def test_change_forecast(self):
        load_p_f = 1.0 * self.env_ref.chronics_handler.real_data.data.load_p_forecast
        load_q_f = 1.0 * self.env_ref.chronics_handler.real_data.data.load_q_forecast
        prod_p_f = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_p_forecast
        prod_v_f = 1.0 * self.env_ref.chronics_handler.real_data.data.prod_v_forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                self.env_name,
                chronics_class=FromNPY,
                test=True,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": self.load_p,
                    "load_q": self.load_q,
                    "prod_p": self.prod_p,
                    "prod_v": self.prod_v,
                    "load_p_forecast": load_p_f,
                    "load_q_forecast": load_q_f,
                    "prod_p_forecast": prod_p_f,
                    "prod_v_forecast": prod_v_f,
                },
            )

        env.chronics_handler.real_data.change_forecasts(
            self.load_p, self.load_q, self.prod_p, self.prod_v
        )  # should not affect anything
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
            sim_obs, *_ = obs.simulate(env.action_space())
            assert np.all(
                prod_p_f[1 + ts, :-1] == sim_obs.gen_p[:-1]
            ), f"error at iteration {ts}"
            assert (
                sim_obs.minute_of_hour == (obs.minute_of_hour + 5) % 60
            ), f"error at iteration {ts}"

        env.reset()  # now forecast should be modified
        for ts in range(10):
            obs, *_ = env.step(env.action_space())
            assert np.all(
                self.prod_p[1 + ts, :-1] == obs.gen_p[:-1]
            ), f"error at iteration {ts}"
            sim_obs, *_ = obs.simulate(env.action_space())
            assert np.all(obs.gen_p == sim_obs.gen_p), f"error at iteration {ts}"
            assert (
                sim_obs.minute_of_hour == (obs.minute_of_hour + 5) % 60
            ), f"error at iteration {ts}"
        env.close()


class TestNPYChronicsWithHazards(unittest.TestCase):
    """
    This class tests the possibility in grid2op to limit the number of call to "obs.simulate"
    """

    def test_maintenance_ok(self):
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        env_path = os.path.join(PATH_DATA_TEST, "env_14_test_maintenance")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_ref = grid2op.make(env_path, test=True, param=param, _add_to_name=type(self).__name__)
        env_ref.chronics_handler.real_data.data.maintenance_starting_hour = 1
        env_ref.chronics_handler.real_data.data.maintenance_ending_hour = 2
        env_ref.seed(0)  # 1 -> 108
        env_ref.reset()
        load_p = 1.0 * env_ref.chronics_handler.real_data.data.load_p
        load_q = 1.0 * env_ref.chronics_handler.real_data.data.load_q
        prod_p = 1.0 * env_ref.chronics_handler.real_data.data.prod_p
        prod_v = 1.0 * env_ref.chronics_handler.real_data.data.prod_v
        load_p_f = 1.0 * env_ref.chronics_handler.real_data.data.load_p_forecast
        load_q_f = 1.0 * env_ref.chronics_handler.real_data.data.load_q_forecast
        prod_p_f = 1.0 * env_ref.chronics_handler.real_data.data.prod_p_forecast
        prod_v_f = 1.0 * env_ref.chronics_handler.real_data.data.prod_v_forecast
        maintenance = copy.deepcopy(env_ref.chronics_handler.real_data.data.maintenance)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                env_path,
                chronics_class=FromNPY,
                test=True,
                _add_to_name=type(self).__name__,
                data_feeding_kwargs={
                    "i_start": 0,
                    "i_end": 10,  # excluded
                    "load_p": load_p,
                    "load_q": load_q,
                    "prod_p": prod_p,
                    "prod_v": prod_v,
                    "load_p_forecast": load_p_f,
                    "load_q_forecast": load_q_f,
                    "prod_p_forecast": prod_p_f,
                    "prod_v_forecast": prod_v_f,
                    "maintenance": maintenance,
                },
                param=param,
            )
        obs = env.reset()
        obs_ref = env_ref.reset()
        for ts in range(8):
            obs, *_ = env.step(env.action_space())
            obs_ref, *_ = env_ref.step(env.action_space())
            assert np.all(
                obs.time_before_cooldown_line == obs_ref.time_before_cooldown_line
            ), f"error at step {ts}"
            sim_obs, *_ = obs.simulate(env.action_space())
            sim_obs_ref, *_ = obs_ref.simulate(env.action_space())
            assert np.all(
                sim_obs.time_before_cooldown_line
                == sim_obs_ref.time_before_cooldown_line
            ), f"error at step {ts}"

    # TODO test obs.max_step
    # test hazards


if __name__ == "__main__":
    unittest.main()
