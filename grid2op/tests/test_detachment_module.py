# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Action import CompleteAction
from grid2op.dtypes import dt_bool, dt_float
from grid2op.Environment.dispatch import DetachmentModule, RedispatchState


class _FakeBackendAction:
    """Only what `DetachmentModule.compute` reads from the backend action."""
    def __init__(self, gen_detached, load_detached):
        self._gen_detached = np.array(gen_detached, dtype=dt_bool)
        self._load_detached = np.array(load_detached, dtype=dt_bool)

    def get_gen_detached(self):
        return self._gen_detached

    def get_load_detached(self):
        return self._load_detached


class TestDetachmentModuleExplicitInputs(unittest.TestCase):
    """`DetachmentModule.compute` only uses the data it is given (no environment needed)."""
    def setUp(self) -> None:
        self.module = DetachmentModule(env=None)
        self.state = RedispatchState.allocate(n_gen=3, n_storage=0, n_load=2)
        self.state.actual_dispatch[:] = [1.0, 2.0, 0.0]
        self.new_p = np.array([10.0, 20.0, 30.0], dtype=dt_float)
        self.new_p_th = np.array([11.0, 20.0, 30.0], dtype=dt_float)
        self.new_load_p = np.array([5.0, 7.0], dtype=dt_float)
        return super().setUp()

    def test_load_detached(self):
        bk_act = _FakeBackendAction([False, False, False], [False, True])
        res = self.module.compute(self.new_p, self.new_p_th, self.new_load_p, self.state, bk_act)
        # the load consumed 7 MW: the generators produce 7 MW less
        assert res.detached_mw == dt_float(-7.0)
        assert self.state.detached_elements_mw == dt_float(-7.0)
        assert self.state.detached_elements_mw_prev == dt_float(-7.0)
        assert not res.gen_detached.any()
        np.testing.assert_array_equal(self.new_p, [10.0, 20.0, 30.0])
        # the load vector is only read
        np.testing.assert_array_equal(self.new_load_p, [5.0, 7.0])

        # next step, still detached: only the variation is compensated
        new_load_p = np.array([5.0, 8.0], dtype=dt_float)
        res = self.module.compute(self.new_p, self.new_p_th, new_load_p, self.state, bk_act)
        assert res.detached_mw == dt_float(-1.0)
        assert self.state.detached_elements_mw_prev == dt_float(-8.0)

    def test_gen_detached(self):
        bk_act = _FakeBackendAction([True, False, False], [False, False])
        res = self.module.compute(self.new_p, self.new_p_th, self.new_load_p, self.state, bk_act)
        # 10 MW of production lost, plus its 1 MW of dispatch
        assert res.detached_mw == dt_float(11.0)
        assert self.state.detached_elements_mw_prev == dt_float(10.0)
        np.testing.assert_array_equal(res.gen_detached, [True, False, False])
        np.testing.assert_array_equal(self.new_p, [0.0, 20.0, 30.0])
        np.testing.assert_array_equal(self.new_p_th, [0.0, 20.0, 30.0])
        np.testing.assert_array_equal(self.state.actual_dispatch, [0.0, 2.0, 0.0])


class TestDetachedLoadUsesTimeSeries(unittest.TestCase):
    """The power of a detached load is the one of the time series, an injection of the
    agent on this load is not taken into account."""
    def _make_env(self, suffix):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage",
                               test=True,
                               allow_detachment=True,
                               action_class=CompleteAction,
                               _add_to_name=type(self).__name__ + suffix)
        return env

    def _step_detach(self, env, load_p_override):
        obs = env.reset(seed=0, options={"time serie id": 1})
        dict_ = {"set_bus": {"loads_id": [(self.load_id, -1)]}}
        if load_p_override:
            load_p = np.full(type(env).n_load, np.nan, dtype=dt_float)
            load_p[self.load_id] = obs.load_p[self.load_id] + 10.0
            dict_["injection"] = {"load_p": load_p}
        obs, _, done, info = env.step(env.action_space(dict_))
        assert not done, info["exception"]
        assert not info["is_ambiguous"]
        assert not info["is_illegal"]
        return obs, env._dispatch_state.detached_elements_mw

    def setUp(self) -> None:
        self.load_id = 3
        self.env_ts = self._make_env("_ts")
        self.env_agent = self._make_env("_agent")
        return super().setUp()

    def tearDown(self) -> None:
        self.env_ts.close()
        self.env_agent.close()
        return super().tearDown()

    def test_agent_injection_ignored(self):
        obs_ts, mw_ts = self._step_detach(self.env_ts, load_p_override=False)
        obs_ag, mw_ag = self._step_detach(self.env_agent, load_p_override=True)
        assert abs(mw_ts) >= 1.0
        assert mw_ag == mw_ts
        assert obs_ag.load_p_detached[self.load_id] == obs_ts.load_p_detached[self.load_id]
        assert obs_ts.load_p_detached[self.load_id] == -mw_ts


if __name__ == "__main__":
    unittest.main()
