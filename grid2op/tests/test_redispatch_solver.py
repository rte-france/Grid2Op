# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Action import CompleteAction
from grid2op.Agent import DoNothingAgent
from grid2op.Environment.dispatch import (BaseRedispatchSolver,
                                          DefaultRedispatchSolver,
                                          RedispatchResult)
from grid2op.Exceptions import EnvError, ImpossibleRedispatching
from grid2op.Runner import Runner


class CountingSolver(BaseRedispatchSolver):
    """A solver that only implements the abstract API: it delegates to the default one
    and counts how many times it has been called."""
    def __init__(self):
        super().__init__()
        self.nb_call = 0
        self.nb_reset = 0
        self._default = DefaultRedispatchSolver()

    def solve(self, constraints, state):
        self.nb_call += 1
        return self._default.solve(constraints, state)

    def reset(self):
        self.nb_reset += 1


class FailingSolver(BaseRedispatchSolver):
    """A solver that never finds a dispatch."""
    def solve(self, constraints, state):
        return RedispatchResult.failed(ImpossibleRedispatching("no dispatch from FailingSolver"),
                                       unserved_mw=1.)


class WrongReturnSolver(BaseRedispatchSolver):
    """A solver that does not return a RedispatchResult."""
    def solve(self, constraints, state):
        return None


class RecordingSolver(BaseRedispatchSolver):
    """Delegates to the default solver and records its inputs and outputs, and whether
    the state was modified during the call."""
    def __init__(self):
        super().__init__()
        self._default = DefaultRedispatchSolver()
        self.calls = []

    def solve(self, constraints, state):
        actual_before = state.actual_dispatch.copy()
        target_before = state.target_dispatch.copy()
        res = self._default.solve(constraints, state)
        state_unchanged = ((state.actual_dispatch == actual_before).all()
                           and (state.target_dispatch == target_before).all())
        self.calls.append((constraints, res, state_unchanged))
        return res


class WritingSolver(BaseRedispatchSolver):
    """A solver that tries to modify the arrays of the constraints (some of them are
    class attributes of the environment)."""
    def solve(self, constraints, state):
        constraints.pmin[:] = 0.
        return RedispatchResult(success=True, actual_dispatch=state.actual_dispatch.copy())


class NoSuperInitSolver(DefaultRedispatchSolver):
    """A solver that does not call `BaseRedispatchSolver.__init__`."""
    def __init__(self, coeff=1.):
        self.coeff = coeff


class TestCustomRedispatchSolver(unittest.TestCase):
    def _make(self, redispatch_solver):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return grid2op.make("l2rpn_case14_sandbox",
                                test=True,
                                redispatch_solver=redispatch_solver,
                                _add_to_name=type(self).__name__)

    def setUp(self) -> None:
        self.solver = CountingSolver()
        self.env = self._make(self.solver)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.gen_id = int(np.nonzero(type(self.env).gen_redispatchable)[0][0])
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _redisp_act(self, env):
        return env.action_space({"redispatch": [(self.gen_id, 1.)]})

    def test_env_uses_a_copy_of_the_solver(self):
        solver = self.env._redispatch_solver
        assert isinstance(solver, CountingSolver)
        # the instance given to make is not modified, each env has its own copy
        assert solver is not self.solver
        assert self.solver.env is None
        assert solver.env is self.env

    def test_step(self):
        solver = self.env._redispatch_solver
        nb_reset = solver.nb_reset
        obs, reward, done, info = self.env.step(self._redisp_act(self.env))
        assert not done, info["exception"]
        assert not info["exception"], info["exception"]
        assert solver.nb_call >= 1
        assert abs(obs.target_dispatch[self.gen_id] - 1.) <= 1e-5
        assert abs(obs.actual_dispatch.sum()) <= 1e-3
        self.env.reset(seed=0, options={"time serie id": 0})
        assert solver.nb_reset == nb_reset + 1

    def test_class_is_accepted(self):
        env = self._make(CountingSolver)
        try:
            assert isinstance(env._redispatch_solver, CountingSolver)
            env.reset(seed=0, options={"time serie id": 0})
            obs, reward, done, info = env.step(self._redisp_act(env))
            assert not done, info["exception"]
        finally:
            env.close()

    def test_invalid_solver(self):
        with self.assertRaises(EnvError):
            self._make(DefaultRedispatchSolver.solve)
        with self.assertRaises(EnvError):
            self._make(int)

    def test_failing_solver_is_game_over(self):
        env = self._make(FailingSolver())
        try:
            env.reset(seed=0, options={"time serie id": 0})
            obs, reward, done, info = env.step(self._redisp_act(env))
            assert done
            assert any("no dispatch from FailingSolver" in str(exc) for exc in info["exception"])
        finally:
            env.close()

    def test_wrong_return_type(self):
        env = self._make(WrongReturnSolver())
        try:
            env.reset(seed=0, options={"time serie id": 0})
            with self.assertRaises(EnvError):
                env.step(self._redisp_act(env))
        finally:
            env.close()

    def test_simulate(self):
        obs_env = self.env.observation_space.obs_env
        assert isinstance(obs_env._redispatch_solver, CountingSolver)
        assert obs_env._redispatch_solver is not self.env._redispatch_solver
        sim_obs, sim_r, sim_d, sim_i = self.obs.simulate(self._redisp_act(self.env))
        assert not sim_d, sim_i["exception"]
        assert obs_env._redispatch_solver.nb_call >= 1

    def test_forecast_env(self):
        for_env = self.obs.get_forecast_env()
        try:
            assert isinstance(for_env._redispatch_solver, CountingSolver)
            for_env.reset()
            for_obs, *_ = for_env.step(self._redisp_act(for_env))
            assert for_env._redispatch_solver.nb_call >= 1
        finally:
            for_env.close()

    def test_copy(self):
        env_cpy = self.env.copy()
        try:
            assert isinstance(env_cpy._redispatch_solver, CountingSolver)
            assert env_cpy._redispatch_solver is not self.env._redispatch_solver
            assert env_cpy._redispatch_solver.env is env_cpy
            assert isinstance(env_cpy.observation_space.obs_env._redispatch_solver, CountingSolver)
        finally:
            env_cpy.close()

    def test_constraints_are_read_only(self):
        env = self._make(WritingSolver())
        try:
            env.reset(seed=0, options={"time serie id": 0})
            pmin_before = type(env).gen_pmin.copy()
            with self.assertRaises(ValueError):
                env.step(self._redisp_act(env))
            assert (type(env).gen_pmin == pmin_before).all()
        finally:
            env.close()

    def test_instance_without_super_init(self):
        env = self._make(NoSuperInitSolver(coeff=2.))
        try:
            assert isinstance(env._redispatch_solver, NoSuperInitSolver)
            assert env._redispatch_solver.coeff == 2.
            assert env._redispatch_solver.env is env
            env.reset(seed=0, options={"time serie id": 0})
            obs, reward, done, info = env.step(self._redisp_act(env))
            assert not done, info["exception"]
        finally:
            env.close()

    def test_runner(self):
        params = self.env.get_params_for_runner()
        assert isinstance(params["redispatch_solver"], CountingSolver)
        assert params["redispatch_solver"].env is None
        runner = Runner(**params, agentClass=DoNothingAgent)
        env_runner, _ = runner._new_env(self.env.parameters)
        try:
            assert isinstance(env_runner._redispatch_solver, CountingSolver)
        finally:
            env_runner.close()
        res = runner.run(nb_episode=1, max_iter=3)
        assert len(res) == 1


class TestGenRedispatchableNotModified(unittest.TestCase):
    """When a detached generator makes the dispatch infeasible, the solver tries to use all
    the redispatchable generators. It used to remove the detached generators from the class
    attribute `gen_redispatchable` (shared by all the envs) while doing so."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        params = self.env.parameters
        params.IGNORE_MIN_UP_DOWN_TIME = True
        params.ALLOW_DISPATCH_GEN_SWITCH_OFF = True
        self.env.change_parameters(params)
        self.env.reset(seed=0, options={"time serie id": 1})
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_gen_redispatchable_unchanged(self):
        cls = type(self.env)
        gen_redisp_before = cls.gen_redispatchable.copy()
        gen_id = int(np.nonzero(gen_redisp_before)[0][0])
        obs, reward, done, info = self.env.step(self.env.action_space(
            {"set_bus": {"generators_id": [(gen_id, -1)]}}
        ))
        # detaching this generator is too much for the ramps of the others
        assert done
        assert any(isinstance(exc, ImpossibleRedispatching) for exc in info["exception"])
        assert (cls.gen_redispatchable == gen_redisp_before).all()
        assert (type(self.env.action_space).gen_redispatchable == gen_redisp_before).all()


class TestSimulateAfterDetachment(unittest.TestCase):
    """obs.simulate must keep compensating a load detached at the previous step (the
    previous detached power was restored from the wrong key and the power of the detached
    loads was not known by the environment used by simulate)."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        self.load_id = 3
        self.env.reset(seed=0, options={"time serie id": 1})
        self.obs, _, done, info = self.env.step(self.env.action_space(
            {"set_bus": {"loads_id": [(self.load_id, -1)]}}
        ))
        assert not done, info["exception"]
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_state_restored(self):
        # one step after the detachment the two values differ
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        params = obs._env_internal_params
        assert abs(params["_detached_elements_mw"] - params["_detached_elements_mw_prev"]) >= 1.
        obs_env = self.env.observation_space.obs_env
        obs_env._reset_to_orig_state(obs)
        assert obs_env._detached_elements_mw == params["_detached_elements_mw"]
        assert obs_env._detached_elements_mw_prev == params["_detached_elements_mw_prev"]

    def test_simulate(self):
        sim_obs, _, sim_done, sim_info = self.obs.simulate(self.env.action_space({}))
        assert not sim_done, sim_info["exception"]
        assert sim_obs.load_detached[self.load_id]
        # the load is still detached, the generators keep compensating it
        # (up to the variation of its forecast)
        assert abs(sim_obs.load_p_detached[self.load_id] - self.obs.load_p_detached[self.load_id]) <= 0.5
        assert abs(sim_obs.actual_dispatch.sum() - self.obs.actual_dispatch.sum()) <= 0.5
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        assert abs(sim_obs.actual_dispatch.sum() - obs.actual_dispatch.sum()) <= 0.5


    def test_forecast_env(self):
        # the forecast env used to start its first (internal) step with the backend of the
        # observation (load detached) but without the state of the dispatch
        for_env = self.obs.get_forecast_env()
        try:
            for_obs = for_env.reset()
            assert for_obs.load_detached[self.load_id]
            assert abs(for_obs.actual_dispatch.sum() - self.obs.actual_dispatch.sum()) <= 1e-3
            for_obs, _, for_done, for_info = for_env.step(for_env.action_space({}))
            assert not for_done, for_info["exception"]
        finally:
            for_env.close()
        obs, _, done, info = self.env.step(self.env.action_space({}))
        assert not done, info["exception"]
        assert abs(for_obs.actual_dispatch.sum() - obs.actual_dispatch.sum()) <= 0.5

class TestLimitStorageWithDetachment(unittest.TestCase):
    """With LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION, the power of the detached elements
    must be taken into account when limiting the storage units, and the storage units can
    only be cancelled, not used to compensate the detachment."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        params = self.env.parameters
        params.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(params)
        self.obs = self.env.reset(seed=0, options={"time serie id": 1})
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _act(self, load_id):
        # discharging the storage units at full power, this is feasible alone
        return self.env.action_space({"set_storage": [(0, -5.), (1, -10.)],
                                      "set_bus": {"loads_id": [(load_id, -1)]}})

    def test_storage_limited(self):
        # detaching load 0 alone is feasible, but not together with the storage units
        obs, _, done, info = self.env.step(self._act(0))
        assert not done, info["exception"]
        assert obs.load_detached[0]
        # the storage units have been limited, not cancelled nor reversed
        assert (obs.storage_power < 0.).all()
        assert (obs.storage_power > [-5., -10.]).all()
        # and the storage units really produce what the dispatch took into account
        losses = obs.gen_p.sum() - obs.load_p.sum() - obs.storage_power.sum()
        assert 0. < losses < 10., f"{losses}"
        cls = type(self.env)
        coeff = self.env.delta_time_seconds / 3600.
        expected = self.obs.storage_charge + obs.storage_power * coeff / cls.storage_discharging_efficiency
        if self.env.parameters.ACTIVATE_STORAGE_LOSS:
            expected -= cls.storage_loss * coeff
        assert np.allclose(obs.storage_charge, expected, atol=1e-3), f"{obs.storage_charge} vs {expected}"

    def test_storage_does_not_compensate_detachment(self):
        # detaching load 2 is infeasible alone, cancelling the storage units cannot help
        obs, _, done, info = self.env.step(self._act(2))
        assert done
        assert any(isinstance(exc, ImpossibleRedispatching) for exc in info["exception"])


class TestInjectionWithoutRedispatchData(unittest.TestCase):
    """The injections of the action override the time series even when the grid has no
    redispatching data and no storage unit."""
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        env_path = os.path.join(self.tmp_dir, "case5_no_redisp")
        src = os.path.join(os.path.dirname(grid2op.__file__), "data", "rte_case5_example")
        shutil.copytree(src, env_path,
                        ignore=shutil.ignore_patterns("prods_charac.csv", "_grid2op_classes", "__pycache__"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(env_path,
                                    test=True,
                                    action_class=CompleteAction,
                                    _add_to_name=type(self).__name__)
        assert not type(self.env).redispatching_unit_commitment_availble
        assert type(self.env).n_storage == 0
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        return super().tearDown()

    def test_load_p_injection(self):
        load_p = self.obs.load_p.copy()
        load_p[0] += 1.
        obs, _, done, info = self.env.step(self.env.action_space({"injection": {"load_p": load_p}}))
        assert not done, info["exception"]
        assert abs(obs.load_p[0] - load_p[0]) <= 1e-4, f"{obs.load_p[0]} vs {load_p[0]}"


class TestRedispatchResult(unittest.TestCase):
    """The solver receives a single power to compensate and returns a result, without
    modifying the state of the environment."""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    test=True,
                                    allow_detachment=True,
                                    action_class=CompleteAction,
                                    redispatch_solver=RecordingSolver(),
                                    _add_to_name=type(self).__name__)
        self.obs = self.env.reset(seed=0, options={"time serie id": 1})
        self.solver = self.env._redispatch_solver
        self.solver.calls.clear()
        return super().setUp()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_success(self):
        obs, _, done, info = self.env.step(self.env.action_space(
            {"set_storage": [(0, 2.)], "set_bus": {"loads_id": [(3, -1)]}}
        ))
        assert not done, info["exception"]
        assert self.solver.calls
        constraints, res, state_unchanged = self.solver.calls[-1]
        assert state_unchanged
        assert res.success
        assert res.exception is None
        assert res.unserved_mw == 0.
        # the env applied the dispatch of the result
        assert np.allclose(obs.actual_dispatch, res.actual_dispatch)
        # one power to compensate, the sum of the contributions (same sign convention)
        assert set(constraints.contributions) == {"storage", "curtailment", "detachment"}
        assert abs(constraints.power_to_compensate_mw - sum(constraints.contributions.values())) <= 1e-6
        # the storage units absorb (the generators produce more), the load is detached
        # (the generators produce less)
        assert constraints.contributions["storage"] > 0.
        assert constraints.contributions["detachment"] < 0.
        assert abs(res.actual_dispatch.sum() - constraints.power_to_compensate_mw) <= 1e-3

    def test_failure_unserved(self):
        # detaching this load asks the generators to decrease more than their ramps allow
        obs, _, done, info = self.env.step(self.env.action_space(
            {"set_bus": {"loads_id": [(2, -1)]}}
        ))
        assert done
        constraints, res, state_unchanged = self.solver.calls[-1]
        assert state_unchanged
        assert not res.success
        assert res.actual_dispatch is None
        assert isinstance(res.exception, ImpossibleRedispatching)
        assert res.exception in info["exception"]
        # the generators cannot decrease enough: negative, and less than what is asked
        assert np.isfinite(res.unserved_mw)
        assert res.unserved_mw < 0.
        assert res.unserved_mw > constraints.power_to_compensate_mw


if __name__ == "__main__":
    unittest.main()
