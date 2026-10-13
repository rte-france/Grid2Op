# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import dataclasses
import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Action import CompleteAction
from grid2op.dtypes import dt_bool, dt_float, dt_int
from grid2op.Environment import BaseEnv
from grid2op.Environment.dispatch import RedispatchState
from grid2op.Environment.dispatch.dispatchTypes import (DISPATCH_FIELD_GROUPS,
                                                        DISPATCH_FIELD_KINDS)

N_GEN, N_STORAGE, N_LOAD = 4, 2, 3
ALL_FIELDS = dataclasses.fields(RedispatchState)


def _other_value(fld, idx=0):
    """A value different from the initial one, of the right dtype."""
    meta = fld.metadata
    if meta["dtype"] is dt_bool:
        return not meta["init"]
    return meta["dtype"](meta["init"] + 2 + idx)


def _filled_state():
    """A state where every entry of every field differs from its initial value."""
    res = RedispatchState.allocate(N_GEN, N_STORAGE, N_LOAD)
    for idx, fld in enumerate(ALL_FIELDS):
        val = _other_value(fld, idx)
        if fld.metadata["kind"] == "scalar":
            setattr(res, fld.name, val)
        else:
            getattr(res, fld.name)[:] = val
    return res


def _assert_field_equal(val1, val2, name):
    if isinstance(val1, np.ndarray):
        assert isinstance(val2, np.ndarray), name
        assert val1.dtype == val2.dtype, name
        np.testing.assert_array_equal(val1, val2, err_msg=name)
    else:
        assert val1 == val2, name


class TestRedispatchStateSpec(unittest.TestCase):
    def test_every_field_has_a_spec(self):
        # fails if a field is added without `_state_field`
        assert len(ALL_FIELDS) > 0
        for fld in ALL_FIELDS:
            meta = fld.metadata
            for key in ("kind", "dtype", "init", "group", "restore", "reset", "reset_mask"):
                assert key in meta, f"{fld.name} has no \"{key}\" in its spec"
            assert meta["kind"] in DISPATCH_FIELD_KINDS, fld.name
            assert meta["group"] in DISPATCH_FIELD_GROUPS, fld.name
            assert meta["dtype"] in (dt_float, dt_int, dt_bool), fld.name

    def test_allocate(self):
        sizes = {"gen": N_GEN, "storage": N_STORAGE, "load": N_LOAD}
        state = RedispatchState.allocate(N_GEN, N_STORAGE, N_LOAD)
        for fld in ALL_FIELDS:
            meta = fld.metadata
            val = getattr(state, fld.name)
            if meta["kind"] == "scalar":
                assert val == meta["init"], fld.name
                assert isinstance(val, meta["dtype"]), fld.name
            else:
                assert val.shape == (sizes[meta["kind"]],), fld.name
                assert val.dtype == meta["dtype"], fld.name
                assert (val == meta["init"]).all(), fld.name

    def test_env_attributes(self):
        # BaseEnv._<field> forwards to the dispatch state
        for fld in ALL_FIELDS:
            assert isinstance(getattr(BaseEnv, f"_{fld.name}", None), property), fld.name

    def test_copy_independent(self):
        state = _filled_state()
        ref = state.to_dict()
        cpy = state.copy()
        for fld in ALL_FIELDS:
            _assert_field_equal(getattr(cpy, fld.name), ref[fld.name], fld.name)
        # modify the copy
        for fld in ALL_FIELDS:
            if fld.metadata["kind"] == "scalar":
                setattr(cpy, fld.name, fld.metadata["dtype"](-1))
            else:
                arr = getattr(cpy, fld.name)
                arr[:] = ~arr if arr.dtype == dt_bool else -1
        for fld in ALL_FIELDS:
            _assert_field_equal(getattr(state, fld.name), ref[fld.name], fld.name)

    def test_reset_per_group(self):
        renewable = np.array([True, False, True, False], dtype=dt_bool)
        masks = {"gen_renewable": renewable}
        for group in DISPATCH_FIELD_GROUPS:
            state = _filled_state()
            ref = state.to_dict()
            state.reset(group, masks=masks)
            for fld in ALL_FIELDS:
                meta = fld.metadata
                val = getattr(state, fld.name)
                if meta["group"] != group or not meta["reset"]:
                    _assert_field_equal(val, ref[fld.name], fld.name)
                elif meta["kind"] == "scalar":
                    assert val == meta["init"], fld.name
                    assert isinstance(val, meta["dtype"]), fld.name
                elif meta["reset_mask"] is not None:
                    mask = masks[meta["reset_mask"]]
                    assert (val[mask] == meta["init"]).all(), fld.name
                    np.testing.assert_array_equal(val[~mask], ref[fld.name][~mask], err_msg=fld.name)
                else:
                    assert (val == meta["init"]).all(), fld.name

    def test_reset_all(self):
        state = _filled_state()
        state.reset(masks={"gen_renewable": np.ones(N_GEN, dtype=dt_bool)})
        init = RedispatchState.allocate(N_GEN, N_STORAGE, N_LOAD)
        for fld in ALL_FIELDS:
            if fld.metadata["reset"]:
                _assert_field_equal(getattr(state, fld.name), getattr(init, fld.name), fld.name)

    def test_reset_mask_required(self):
        state = _filled_state()
        with self.assertRaises(KeyError):
            state.reset("curtailment")
        with self.assertRaises(ValueError):
            state.reset("unknown group")

    def test_dict_round_trip(self):
        state = _filled_state()
        dict_ = state.to_dict()
        assert set(dict_) == {fld.name for fld in ALL_FIELDS}
        # the dictionary does not share its arrays with the state
        state.actual_dispatch[:] = -10.
        assert (dict_["actual_dispatch"] != -10.).all()

        other = RedispatchState.allocate(N_GEN, N_STORAGE, N_LOAD)
        init = other.to_dict()
        other.load_from(dict_)
        for fld in ALL_FIELDS:
            expected = dict_ if fld.metadata["restore"] else init
            _assert_field_equal(getattr(other, fld.name), expected[fld.name], fld.name)


class TestRedispatchStateInEnv(unittest.TestCase):
    """The state is copied with the environment and restored from the observation."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        self.env.reset(seed=0, options={"time serie id": 0})
        renew_id = int(np.nonzero(type(self.env).gen_renewable)[0][0])
        self.actions = [
            {"set_storage": [(0, 3.0)], "curtail": [(renew_id, 0.2)], "redispatch": [(0, 1.0)]},
            {"set_bus": {"loads_id": [(3, -1)]}, "set_storage": [(1, -2.0)]},
            {},
        ]
        for act in self.actions:
            self.obs, _, done, info = self.env.step(self.env.action_space(act))
            assert not done, info["exception"]
        # the scenario uses every part of the state
        state = self.env._dispatch_state
        assert (state.storage_current_charge != state.storage_previous_charge).any()
        assert (state.limit_curtailment != 1.0).any()
        assert abs(state.detached_elements_mw_prev) >= 1.0
        assert (np.abs(state.actual_dispatch) >= 1.0).any()
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _assert_same_state(self, state1, state2, only_restored=False):
        for fld in ALL_FIELDS:
            if only_restored and not fld.metadata["restore"]:
                continue
            _assert_field_equal(getattr(state1, fld.name), getattr(state2, fld.name), fld.name)

    def test_env_copy(self):
        env_cpy = self.env.copy()
        try:
            self._assert_same_state(env_cpy._dispatch_state, self.env._dispatch_state)
            for fld in ALL_FIELDS:
                val = getattr(env_cpy._dispatch_state, fld.name)
                if isinstance(val, np.ndarray):
                    assert not np.shares_memory(val, getattr(self.env._dispatch_state, fld.name)), fld.name
            # both evolve the same way
            for act in self.actions:
                obs1, *_ = self.env.step(self.env.action_space(act))
                obs2, *_ = env_cpy.step(env_cpy.action_space(act))
                assert obs1 == obs2
            self._assert_same_state(env_cpy._dispatch_state, self.env._dispatch_state)
        finally:
            env_cpy.close()

    def test_restore_from_obs(self):
        obs_env = self.env.observation_space.obs_env
        obs_env._reset_to_orig_state(self.obs)
        self._assert_same_state(obs_env._dispatch_state, self.env._dispatch_state, only_restored=True)
        np.testing.assert_array_equal(obs_env._actual_dispatch, self.obs.actual_dispatch)
        np.testing.assert_array_equal(obs_env._target_dispatch, self.obs.target_dispatch)
        np.testing.assert_array_equal(obs_env._storage_current_charge, self.obs.storage_charge)
        np.testing.assert_array_equal(obs_env._storage_power, self.obs.storage_power)
        np.testing.assert_array_equal(obs_env._action_storage, self.obs.storage_power_target)
        np.testing.assert_array_equal(obs_env._limit_curtailment, self.obs.curtailment_limit)
        np.testing.assert_array_equal(obs_env._gen_before_curtailment, self.obs.gen_p_before_curtail)

    def test_obs_copy_keeps_state(self):
        # the state saved in the observation is not modified by the next steps
        saved = self.obs._env_internal_params["_dispatch_state"]
        ref = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in saved.items()}
        for act in self.actions:
            self.env.step(self.env.action_space(act))
        for fld in ALL_FIELDS:
            _assert_field_equal(saved[fld.name], ref[fld.name], fld.name)


if __name__ == "__main__":
    unittest.main()
