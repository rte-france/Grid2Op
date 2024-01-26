# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from os import PathLike
from typing import Optional, Union
import warnings
import unittest
from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Environment import MaskedEnvironment, TimedOutEnvironment
from grid2op.Runner import Runner
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Exceptions import Grid2OpException, EnvError
import pdb


class _AuxFakeBackendSupport(PandaPowerBackend):
    def cannot_handle_more_than_2_busbar(self):
        """dont do it at home !"""
        return self.can_handle_more_than_2_busbar()


class _AuxFakeBackendNoSupport(PandaPowerBackend):
    def can_handle_more_than_2_busbar(self):
        """dont do it at home !"""
        return self.cannot_handle_more_than_2_busbar()


class _AuxFakeBackendNoCalled(PandaPowerBackend):
    def can_handle_more_than_2_busbar(self):
        """dont do it at home !"""
        pass
    def cannot_handle_more_than_2_busbar(self):
        """dont do it at home !"""
        pass


class TestRightNumber(unittest.TestCase):
    """This test that, when changing n_busbar in make it is 
    back propagated where it needs"""
    def _aux_fun_test(self, env, n_busbar):
        assert type(env).n_busbar_per_sub == n_busbar, f"type(env).n_busbar_per_sub = {type(env).n_busbar_per_sub} != {n_busbar}"
        assert type(env.backend).n_busbar_per_sub == n_busbar, f"env.backend).n_busbar_per_sub = {type(env.backend).n_busbar_per_sub} != {n_busbar}"
        assert type(env.action_space).n_busbar_per_sub == n_busbar, f"type(env.action_space).n_busbar_per_sub = {type(env.action_space).n_busbar_per_sub} != {n_busbar}"
        assert type(env.observation_space).n_busbar_per_sub == n_busbar, f"type(env.observation_space).n_busbar_per_sub = {type(env.observation_space).n_busbar_per_sub} != {n_busbar}"
        obs = env.reset(seed=0, options={"time serie id": 0})
        assert type(obs).n_busbar_per_sub == n_busbar, f"type(obs).n_busbar_per_sub = {type(obs).n_busbar_per_sub} != {n_busbar}"
        act = env.action_space()
        assert type(act).n_busbar_per_sub == n_busbar, f"type(act).n_busbar_per_sub = {type(act).n_busbar_per_sub} != {n_busbar}"
        
    def test_fail_if_not_int(self):
        with self.assertRaises(Grid2OpException):
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar="froiy", _add_to_name=type(self).__name__+"_wrong")
        with self.assertRaises(Grid2OpException):
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3.5, _add_to_name=type(self).__name__+"_wrong")
            
    def test_regular_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
    def test_multimix_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
    def test_masked_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_mask_2"),
                                    lines_of_interest=np.ones(shape=20, dtype=bool))
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_mask_3"),
                                    lines_of_interest=np.ones(shape=20, dtype=bool))
        self._aux_fun_test(env, 3)
        
    def test_to_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_to_2"),
                                      time_out_ms=3000)
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_to_3"),
                                      time_out_ms=3000)
        self._aux_fun_test(env, 3)
    
    def test_xxxhandle_more_than_2_busbar_not_called(self):
        """when using a backend that did not called the `can_handle_more_than_2_busbar_not_called`
        nor the `cannot_handle_more_than_2_busbar_not_called` then it's equivalent 
        to not support this new feature."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoCalled(), test=True, _add_to_name=type(self).__name__+"_nocall_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoCalled(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_nocall_3")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
    
    def test_cannot_handle_more_than_2_busbar_not_called(self):
        """when using a backend that called `cannot_handle_more_than_2_busbar_not_called` then it's equivalent 
        to not support this new feature."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, _add_to_name=type(self).__name__+"_dontcalled_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_dontcalled_3")
        self._aux_fun_test(env, 2)
        
    def test_env_copy(self):
        """test env copy does work correctly"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_copy_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        env_cpy = env.copy()
        self._aux_fun_test(env_cpy, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_copy_3")
        self._aux_fun_test(env, 3)
        env_cpy = env.copy()
        self._aux_fun_test(env_cpy, 3)
        
    def test_two_env_same_name(self):
        """test i can load 2 env with the same name but different n_busbar"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_2 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_same_name")
        self._aux_fun_test(env_2, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_3 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_same_name")
        self._aux_fun_test(env_3, 3)  # check env_3 has indeed 3 buses
        self._aux_fun_test(env_2, DEFAULT_N_BUSBAR_PER_SUB)  # check env_2 is not modified


class _TestAgentRightNBus(BaseAgent):
    def __init__(self, action_space: ActionSpace, nb_bus : int):
        super().__init__(action_space)
        self.nb_bus = nb_bus
        assert type(self.action_space).n_busbar_per_sub == self.nb_bus
    
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        assert type(observation).n_busbar_per_sub == self.nb_bus
        return self.action_space()
    
    
class TestRunner(unittest.TestCase):
    """Testthe runner is compatible with the feature"""
    def test_single_process(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # 3 busbars as asked
            env_3 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # 2 busbars only because backend does not support it
            env_2 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_2")
        
        agent_3 = _TestAgentRightNBus(env_3.action_space, 3)
        agent_2 = _TestAgentRightNBus(env_2.action_space, 2)
        
        runner_3 = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_3)
        res = runner_3.run(nb_episode=1, max_iter=5)
        
        runner_2 = Runner(**env_2.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
        res = runner_2.run(nb_episode=1, max_iter=5)
        
        with self.assertRaises(AssertionError):
            runner_3_ko = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
            res = runner_3_ko.run(nb_episode=1, max_iter=5)
            
        with self.assertRaises(AssertionError):
            runner_2_ko = Runner(**env_2.get_params_for_runner(), agentClass=None, agentInstance=agent_3)
            res = runner_2_ko.run(nb_episode=1, max_iter=5)
    
    def test_two_env_same_name(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_2 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_same_name")
        
        agent_2 = _TestAgentRightNBus(env_2.action_space, 2)
        runner_2 = Runner(**env_2.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
        res = runner_2.run(nb_episode=1, max_iter=5)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_3 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_same_name") 
        agent_3 = _TestAgentRightNBus(env_3.action_space, 3)
        runner_3 = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_3)
        res = runner_3.run(nb_episode=1, max_iter=5)
        
        with self.assertRaises(AssertionError):
            runner_3_ko = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
            res = runner_3_ko.run(nb_episode=1, max_iter=5)
            
        with self.assertRaises(AssertionError):
            runner_2_ko = Runner(**env_2.get_params_for_runner(), agentClass=None, agentInstance=agent_3)
            res = runner_2_ko.run(nb_episode=1, max_iter=5)
            
    def test_two_process(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # 3 busbars as asked
            env_3 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3_twocores")
            
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # 2 busbars only because backend does not support it
            env_2 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_2_twocores")
        
        agent_3 = _TestAgentRightNBus(env_3.action_space, 3)
        agent_2 = _TestAgentRightNBus(env_2.action_space, 2)
        
        runner_3 = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_3)
        res = runner_3.run(nb_episode=2, nb_process=2, max_iter=5)
        
        runner_2 = Runner(**env_2.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
        res = runner_2.run(nb_episode=2, nb_process=2, max_iter=5)

        # with self.assertRaises(multiprocessing.pool.RemoteTraceback):
        with self.assertRaises(AssertionError):
            runner_3_ko = Runner(**env_3.get_params_for_runner(), agentClass=None, agentInstance=agent_2)
            res = runner_3_ko.run(nb_episode=2, nb_process=2, max_iter=5)


class TestGridObjt(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    backend=_AuxFakeBackendSupport(),
                                    test=True,
                                    n_busbar=3,
                                    _add_to_name=type(self).__name__)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_global_bus_to_local_int(self):
        cls_env = type(self.env)
        # easy case: everything on bus 1
        res = cls_env.global_bus_to_local_int(cls_env.gen_to_subid[0], cls_env.gen_to_subid[0])
        assert res == 1
        
        # bit less easy: one generator is disconnected
        gen_off = 2
        res = cls_env.global_bus_to_local_int(-1, cls_env.gen_to_subid[gen_off])
        assert res == -1
        
        # still a bit more complex: one gen on busbar 2
        gen_on_2 = 3
        res = cls_env.global_bus_to_local_int(cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub, cls_env.gen_to_subid[gen_on_2])
        assert res == 2
        
        # and now a generator on busbar 3
        gen_on_3 = 4
        res = cls_env.global_bus_to_local_int(cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub, cls_env.gen_to_subid[gen_on_3])
        assert res == 3
        
        with self.assertRaises(EnvError):
            gen_on_4 = 4
            res = cls_env.global_bus_to_local_int(cls_env.gen_to_subid[gen_on_4] + 3 * cls_env.n_sub, cls_env.gen_to_subid[gen_on_4])
            
    def test_global_bus_to_local(self):
        cls_env = type(self.env)
        # easy case: everything on bus 1
        res = cls_env.global_bus_to_local(cls_env.gen_to_subid, cls_env.gen_to_subid)
        assert (res == np.ones(cls_env.n_gen, dtype=int)).all()
        
        # bit less easy: one generator is disconnected
        gen_off = 2
        inp_vect = 1 * cls_env.gen_to_subid
        inp_vect[gen_off] = -1
        res = cls_env.global_bus_to_local(inp_vect, cls_env.gen_to_subid)
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_off] = -1
        assert (res == vect).all()
        
        # still a bit more complex: one gen on busbar 2
        gen_on_2 = 3
        inp_vect = 1 * cls_env.gen_to_subid
        inp_vect[gen_on_2] = cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub
        res = cls_env.global_bus_to_local(inp_vect, cls_env.gen_to_subid)
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_on_2] = 2
        assert (res == vect).all()
        
        # and now a generator on busbar 3
        gen_on_3 = 4
        inp_vect = 1 * cls_env.gen_to_subid
        inp_vect[gen_on_3] = cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub
        res = cls_env.global_bus_to_local(inp_vect, cls_env.gen_to_subid)
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_on_3] = 3
        assert (res == vect).all()
        
        # and now we mix all
        inp_vect = 1 * cls_env.gen_to_subid
        inp_vect[gen_off] = -1
        inp_vect[gen_on_2] = cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub
        inp_vect[gen_on_3] = cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub
        res = cls_env.global_bus_to_local(inp_vect, cls_env.gen_to_subid)
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_off] = -1
        vect[gen_on_2] = 2
        vect[gen_on_3] = 3
        assert (res == vect).all()
    
    
    def test_local_bus_to_global_int(self):
        cls_env = type(self.env)
        # easy case: everything on bus 1
        res = cls_env.local_bus_to_global_int(1, cls_env.gen_to_subid[0])
        assert res == cls_env.gen_to_subid[0]
        
        # bit less easy: one generator is disconnected
        gen_off = 2
        res = cls_env.local_bus_to_global_int(-1, cls_env.gen_to_subid[gen_off])
        assert res == -1
        
        # still a bit more complex: one gen on busbar 2
        gen_on_2 = 3
        res = cls_env.local_bus_to_global_int(2, cls_env.gen_to_subid[gen_on_2])
        assert res == cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub
        
        # and now a generator on busbar 3
        gen_on_3 = 4
        res = cls_env.local_bus_to_global_int(3, cls_env.gen_to_subid[gen_on_3])
        assert res == cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub
    
    def test_local_bus_to_global(self):
        cls_env = type(self.env)
        # easy case: everything on bus 1
        res = cls_env.local_bus_to_global(np.ones(cls_env.n_gen, dtype=int), cls_env.gen_to_subid)
        assert (res == cls_env.gen_to_subid).all()
        
        # bit less easy: one generator is disconnected
        gen_off = 2
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_off] = -1
        res = cls_env.local_bus_to_global(vect, cls_env.gen_to_subid)
        assert (res == cls_env.gen_to_subid).sum() == cls_env.n_gen - 1
        assert res[gen_off] == -1
        
        # still a bit more complex: one gen on busbar 2
        gen_on_2 = 3
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_on_2] = 2
        res = cls_env.local_bus_to_global(vect, cls_env.gen_to_subid)
        assert (res == cls_env.gen_to_subid).sum() == cls_env.n_gen - 1
        assert res[gen_on_2] == cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub
        
        # and now a generator on busbar 3
        gen_on_3 = 4
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_on_3] = 3
        res = cls_env.local_bus_to_global(vect, cls_env.gen_to_subid)
        assert (res == cls_env.gen_to_subid).sum() == cls_env.n_gen - 1
        assert res[gen_on_3] == cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub
        
        # and now we mix all
        vect = np.ones(cls_env.n_gen, dtype=int)
        vect[gen_off] = -1
        vect[gen_on_2] = 2
        vect[gen_on_3] = 3
        res = cls_env.local_bus_to_global(vect, cls_env.gen_to_subid)
        assert res[gen_off] == -1
        assert res[gen_on_2] == cls_env.gen_to_subid[gen_on_2] + cls_env.n_sub
        assert res[gen_on_3] == cls_env.gen_to_subid[gen_on_3] + 2 * cls_env.n_sub
    
if __name__ == "__main__":
    unittest.main()
        