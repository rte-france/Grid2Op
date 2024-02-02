# Copyright (c) 2024, RTE (https://www.rte-france.com)
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
from grid2op.Agent import BaseAgent
from grid2op.Environment import MaskedEnvironment, TimedOutEnvironment
from grid2op.Runner import Runner
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB
from grid2op.Action import ActionSpace, BaseAction, CompleteAction
from grid2op.Observation import BaseObservation
from grid2op.Exceptions import Grid2OpException, EnvError, IllegalAction
import pdb


HAS_TIME_AND_MEMORY = False  # test on a big computer only with lots of RAM, and lots of time available...


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
    back propagated where it needs in the class attribute (this includes 
    testing that the observation_space, action_space, runner, environment etc.
    are all 'informed' about this feature)
    
    This class also tests than when the implementation of the backend does not 
    use the new `can_handle_more_than_2_busbar` or `cannot_handle_more_than_2_busbar`
    then the legacy behaviour is used (only 2 busbar per substation even if the 
    user asked for a different number)
    """
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
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar="froiy", _add_to_name=type(self).__name__+"_wrong_str")
        with self.assertRaises(Grid2OpException):
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3.5, _add_to_name=type(self).__name__+"_wrong_float")
            
    def test_regular_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 1)
        
    def test_multimix_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 1)
        
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
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_mask_1"),
                                    lines_of_interest=np.ones(shape=20, dtype=bool))
        self._aux_fun_test(env, 1)
        
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
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = TimedOutEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_to_1"),
                                      time_out_ms=3000)
        self._aux_fun_test(env, 1)
    
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
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoCalled(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_nocall_1")
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
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_dontcalled_1")
        self._aux_fun_test(env, DEFAULT_N_BUSBAR_PER_SUB)
        
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
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_copy_1")
        self._aux_fun_test(env, 1)
        env_cpy = env.copy()
        self._aux_fun_test(env_cpy, 1)
        
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
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_1 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=1, _add_to_name=type(self).__name__+"_same_name")
        self._aux_fun_test(env_1, 1)  # check env_1 has indeed 3 buses
        self._aux_fun_test(env_3, 3)  # check env_3 is not modified
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
    """Test that the GridObj class is fully compatible with this feature"""
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
        """test the function :func:`grid2op.Space.GridObjects.global_bus_to_local_int` """
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
        """test the function :func:`grid2op.Space.GridObjects.global_bus_to_local` """
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
        """test the function :func:`grid2op.Space.GridObjects.local_bus_to_global_int` """
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
        """test the function :func:`grid2op.Space.GridObjects.local_bus_to_global` """
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
    
    
class TestAction_3busbars(unittest.TestCase):
    """This class test the Agent can perform actions (and that actions are properly working)
    even if there are 3 busbars per substation
    """
    def get_nb_bus(self):
        return 3
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage",
                                    backend=_AuxFakeBackendSupport(),
                                    action_class=CompleteAction,
                                    test=True,
                                    n_busbar=self.get_nb_bus(),
                                    _add_to_name=type(self).__name__ + f'_{self.get_nb_bus()}')
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def _aux_test_act_consistent_as_dict(self, act_as_dict, name_xxx, el_id, bus_val):
        if name_xxx is not None:
            # regular element in the topo_vect
            assert "set_bus_vect" in act_as_dict
            tmp = act_as_dict["set_bus_vect"]
            assert len(tmp['modif_subs_id']) == 1
            sub_id = tmp['modif_subs_id'][0]
            assert name_xxx[el_id] in tmp[sub_id]
            assert tmp[sub_id][name_xxx[el_id]]["new_bus"] == bus_val
        else:
            # el not in topo vect (eg shunt)
            assert "shunt" in act_as_dict
            tmp = act_as_dict["shunt"]["shunt_bus"]
            assert tmp[el_id] == bus_val

    def _aux_test_act_consistent_as_serializable_dict(self, act_as_dict, el_nms, el_id, bus_val):
        if el_nms is not None:
            # regular element
            assert "set_bus" in act_as_dict
            assert el_nms in act_as_dict["set_bus"]
            tmp = act_as_dict["set_bus"][el_nms]
            assert tmp == [(el_id, bus_val)]
        else:
            # shunts of other things not in the topo vect
            assert "shunt" in act_as_dict
            tmp = act_as_dict["shunt"]["shunt_bus"]
            assert tmp == [(el_id, bus_val)]
    
    def _aux_test_action(self, act : BaseAction, name_xxx, el_id, bus_val, el_nms):
        assert act.can_affect_something()
        assert not act.is_ambiguous()[0]
        tmp = f"{act}"  # test the print does not crash
        tmp = act.as_dict()  # test I can convert to dict
        self._aux_test_act_consistent_as_dict(tmp, name_xxx, el_id, bus_val)
        tmp = act.as_serializable_dict()  # test I can convert to another type of dict
        self._aux_test_act_consistent_as_serializable_dict(tmp, el_nms, el_id, bus_val)
        
    def _aux_test_set_bus_onebus(self, nm_prop, el_id, bus_val, name_xxx, el_nms):
        act = self.env.action_space()
        setattr(act, nm_prop, [(el_id, bus_val)])
        self._aux_test_action(act, name_xxx, el_id, bus_val, el_nms)
        
    def test_set_load_bus(self):
        self._aux_test_set_bus_onebus("load_set_bus", 0, -1, type(self.env).name_load, 'loads_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus("load_set_bus", 0, bus + 1, type(self.env).name_load, 'loads_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.load_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
            
    def test_set_gen_bus(self):
        self._aux_test_set_bus_onebus("gen_set_bus", 0, -1, type(self.env).name_gen, 'generators_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus("gen_set_bus", 0, bus + 1, type(self.env).name_gen, 'generators_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.gen_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
    
    def test_set_storage_bus(self):
        self._aux_test_set_bus_onebus("storage_set_bus", 0, -1, type(self.env).name_storage, 'storages_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus("storage_set_bus", 0, bus + 1, type(self.env).name_storage, 'storages_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.storage_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
    
    def test_set_lineor_bus(self):
        self._aux_test_set_bus_onebus("line_or_set_bus", 0, -1, type(self.env).name_line, 'lines_or_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus("line_or_set_bus", 0, bus + 1, type(self.env).name_line, 'lines_or_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.line_or_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
            
    def test_set_lineex_bus(self):
        self._aux_test_set_bus_onebus("line_ex_set_bus", 0, -1, type(self.env).name_line, 'lines_ex_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus("line_ex_set_bus", 0, bus + 1, type(self.env).name_line, 'lines_ex_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
    
    def _aux_test_set_bus_onebus_sub_setbus(self, nm_prop, sub_id, el_id_sub, bus_val, name_xxx, el_nms):
        # for now works only with lines_ex (in other words, the name_xxx and name_xxx should be 
        # provided by the user and it's probably not a good idea to use something
        # else than type(self.env).name_line and lines_ex_id
        act = self.env.action_space()
        buses_val = np.zeros(type(self.env).sub_info[sub_id], dtype=int)
        buses_val[el_id_sub] = bus_val
        setattr(act, nm_prop, [(sub_id, buses_val)])
        el_id_in_topo_vect = np.where(act._set_topo_vect == bus_val)[0][0]
        el_type = np.where(type(self.env).grid_objects_types[el_id_in_topo_vect][1:] != -1)[0][0]
        el_id = type(self.env).grid_objects_types[el_id_in_topo_vect][el_type + 1]
        self._aux_test_action(act, name_xxx, el_id, bus_val, el_nms)
        
    def test_sub_set_bus(self):
        self._aux_test_set_bus_onebus_sub_setbus("sub_set_bus", 1, 0, -1, type(self.env).name_line, 'lines_ex_id')
        for bus in range(type(self.env).n_busbar_per_sub):
            self._aux_test_set_bus_onebus_sub_setbus("sub_set_bus", 1, 0, bus + 1, type(self.env).name_line, 'lines_ex_id')
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act.line_ex_set_bus = [(0, type(self.env).n_busbar_per_sub + 1)]
            
    def test_change_deactivated(self):
        assert "set_bus" in type(self.env.action_space()).authorized_keys
        assert self.env.action_space.supports_type("set_bus")
        
        assert "change_bus" not in type(self.env.action_space()).authorized_keys
        assert not self.env.action_space.supports_type("change_bus")
    
    def _aux_test_action_shunt(self, act : BaseAction, el_id, bus_val):
        name_xxx = None
        el_nms = None
        # self._aux_test_action(act, type(self.env).name_shunt, el_id, bus_val, None)  # does not work for a lot of reasons
        assert not act.is_ambiguous()[0]
        tmp = f"{act}"  # test the print does not crash
        tmp = act.as_dict()  # test I can convert to dict
        self._aux_test_act_consistent_as_dict(tmp, name_xxx, el_id, bus_val)
        tmp = act.as_serializable_dict()  # test I can convert to another type of dict
        self._aux_test_act_consistent_as_serializable_dict(tmp, el_nms, el_id, bus_val)
        
    def test_shunt(self):
        el_id = 0
        bus_val = -1
        act = self.env.action_space({"shunt": {"set_bus": [(el_id, bus_val)]}})
        self._aux_test_action_shunt(act, el_id, bus_val)
        
        for bus_val in range(type(self.env).n_busbar_per_sub):
            act = self.env.action_space({"shunt": {"set_bus": [(el_id, bus_val + 1)]}})
            self._aux_test_action_shunt(act, el_id, bus_val + 1)
            
        act = self.env.action_space()
        with self.assertRaises(IllegalAction):
            act = self.env.action_space({"shunt": {"set_bus": [(el_id, type(self.env).n_busbar_per_sub + 1)]}})


class TestAction_1busbar(TestAction_3busbars):
    """This class test the Agent can perform actions (and that actions are properly working)
    even if there is only 1 busbar per substation
    """
    def get_nb_bus(self):
        return 1


class TestActionSpace(unittest.TestCase):
    """This function test the action space, basically the counting 
    of unique possible topologies per substation
    """
    def get_nb_bus(self):
        return 3
    
    def get_env_nm(self):
        return "educ_case14_storage"
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.get_env_nm(),
                                    backend=_AuxFakeBackendSupport(),
                                    action_class=CompleteAction,
                                    test=True,
                                    n_busbar=self.get_nb_bus(),
                                    _add_to_name=type(self).__name__ + f'_{self.get_nb_bus()}')
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_legacy_all_unitary_topologies_set_behaviour(self):
        """make sure nothing broke for 2 busbars per substation even if the implementation changes"""
        class SubMe(TestActionSpace):
            def get_nb_bus(self):
                return 2
            
        tmp = SubMe()
        tmp.setUp()
        res = tmp.env.action_space.get_all_unitary_topologies_set(tmp.env.action_space, _count_only=True)
        res_noalone = tmp.env.action_space.get_all_unitary_topologies_set(tmp.env.action_space,
                                                                          add_alone_line=False,
                                                                          _count_only=True)
        tmp.tearDown()
        assert res == [3, 29, 5, 31, 15, 113, 4, 0, 15, 3, 3, 3, 7, 3], f"found: {res}"
        assert res_noalone == [0, 25, 3, 26, 11, 109, 0, 0, 11, 0, 0, 0, 4, 0], f"found: {res_noalone}"
        
        class SubMe2(TestActionSpace):
            def get_nb_bus(self):
                return 2
            def get_env_nm(self):
                return "l2rpn_idf_2023"
        tmp2 = SubMe2()
        tmp2.setUp()
        res = tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,  _count_only=True)
        res_noalone = tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,
                                                                           add_alone_line=False,
                                                                           _count_only=True)
        tmp2.tearDown()
        assert res == [3, 3, 7, 9, 16, 3, 3, 13, 2, 0, 57, 253, 3, 3, 241, 3, 63, 5, 29, 3, 
                       3, 3, 29, 7, 7, 3, 57, 3, 3, 8, 7, 31, 3, 29, 3, 3, 32, 4, 3, 29, 3, 
                       113, 3, 3, 13, 13, 7, 3, 65505, 3, 7, 3, 3, 125, 13, 497, 3, 3, 505, 
                       13, 15, 57, 2, 4, 15, 61, 3, 8, 63, 121, 4, 3, 0, 3, 31, 5, 1009, 3, 
                       3, 1017, 2, 7, 13, 3, 61, 3, 0, 3, 63, 25, 3, 253, 3, 31, 3, 61, 3, 
                       3, 3, 2033, 3, 3, 15, 13, 61, 7, 5, 3, 3, 15, 0, 0, 9, 3, 3, 0, 0, 3], f"found: {res}"
        assert res_noalone == [0, 0, 4, 7, 11, 0, 0, 10, 0, 0, 53, 246, 0, 0, 236, 0, 57, 3, 
                               25, 0, 0, 0, 25, 4, 4, 0, 53, 0, 0, 4, 4, 26, 0, 25, 0, 0, 26, 
                               0, 0, 25, 0, 109, 0, 0, 10, 10, 4, 0, 65493, 0, 4, 0, 0, 119, 
                               10, 491, 0, 0, 498, 10, 11, 53, 0, 0, 11, 56, 0, 4, 57, 116, 
                               0, 0, 0, 0, 26, 3, 1002, 0, 0, 1009, 0, 4, 10, 0, 56, 0, 0, 
                               0, 57, 22, 0, 246, 0, 26, 0, 56, 0, 0, 0, 2025, 0, 0, 11, 10, 
                               56, 4, 3, 0, 0, 11, 0, 0, 7, 0, 0, 0, 0, 0], f"found: {res_noalone}"

    def test_is_ok_symmetry(self):
        """test the :func:`grid2op.Action.SerializableActionSpace._is_ok_symmetry`"""
        ok = np.array([1, 1, 1, 1])
        assert type(self.env.action_space)._is_ok_symmetry(2, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 1, 1])
        assert type(self.env.action_space)._is_ok_symmetry(2, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 3, 1])
        assert type(self.env.action_space)._is_ok_symmetry(3, ok), f"should not break for {ok}"
        ok = np.array([1, 1, 2, 3])
        assert type(self.env.action_space)._is_ok_symmetry(3, ok), f"should not break for {ok}"
        ok = np.array([1, 1, 2, 2])
        assert type(self.env.action_space)._is_ok_symmetry(4, ok), f"should not break for {ok}"
        
        ko = np.array([1, 3, 2, 1])  # relabel 3 -> 2, so this topology is not valid
        assert not type(self.env.action_space)._is_ok_symmetry(3, ko), f"should break for {ko}"
        ko = np.array([1, 1, 3, 2])  # relabel 3 -> 2, so this topology is not valid
        assert not type(self.env.action_space)._is_ok_symmetry(3, ko), f"should break for {ko}"
        
        ko = np.array([1, 3, 2, 1])  # relabel 3 -> 2, so this topology is not valid
        assert not type(self.env.action_space)._is_ok_symmetry(4, ko), f"should break for {ko}"
        ko = np.array([1, 1, 3, 2])  # relabel 3 -> 2, so this topology is not valid
        assert not type(self.env.action_space)._is_ok_symmetry(4, ko), f"should break for {ko}"

    def test_is_ok_line(self):
        """test the :func:`grid2op.Action.SerializableActionSpace._is_ok_line`"""
        lines_id = np.array([1, 3])
        n_busbar_per_sub = 2
        ok = np.array([1, 1, 1, 1])
        assert type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ok, lines_id), f"should not break for {ok}"
        ok = np.array([1, 2, 2, 1])
        assert type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ok, lines_id), f"should not break for {ok}"
        ko = np.array([1, 2, 1, 2])  # no lines on bus 1
        assert not type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ko, lines_id), f"should break for {ko}"
        
        n_busbar_per_sub = 3  # should have no impact
        ok = np.array([1, 1, 1, 1])
        assert type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ok, lines_id), f"should not break for {ok}"
        ok = np.array([1, 2, 2, 1])
        assert type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ok, lines_id), f"should not break for {ok}"
        ko = np.array([1, 2, 1, 2])  # no lines on bus 1
        assert not type(self.env.action_space)._is_ok_line(n_busbar_per_sub, ko, lines_id), f"should break for {ko}"
        
    def test_2_obj_per_bus(self):
        """test the :func:`grid2op.Action.SerializableActionSpace._is_ok_2`"""
        n_busbar_per_sub = 2
        ok = np.array([1, 1, 1, 1])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 2, 1])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 1, 2])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        
        ko = np.array([1, 2, 2, 2])  # only 1 element on bus 1
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"
        ko = np.array([1, 2, 1, 1])  # only 1 element on bus 2
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"
        ko = np.array([1, 1, 2, 2, 3])  # only 1 element on bus 3
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"
        
        n_busbar_per_sub = 3
        ok = np.array([1, 1, 1, 1])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 2, 1])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        ok = np.array([1, 2, 1, 2])
        assert type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ok), f"should not break for {ok}"
        
        ko = np.array([1, 2, 2, 2])  # only 1 element on bus 1
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"
        ko = np.array([1, 2, 1, 1])  # only 1 element on bus 2
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"
        ko = np.array([1, 1, 2, 2, 3])  # only 1 element on bus 3
        assert not type(self.env.action_space)._is_ok_2(n_busbar_per_sub, ko), f"should break for {ko}"

    def test_1_busbar(self):
        """test :func:`grid2op.Action.SerializableActionSpace.get_all_unitary_topologies_set` 
        when there are only 1 busbar per substation"""
        class SubMe(TestActionSpace):
            def get_nb_bus(self):
                return 1
            
        tmp = SubMe()
        tmp.setUp()
        res = [len(tmp.env.action_space.get_all_unitary_topologies_set(tmp.env.action_space,
                                                                       sub_id))
               for sub_id in range(type(tmp.env).n_sub)]
        res_noalone = [len(tmp.env.action_space.get_all_unitary_topologies_set(tmp.env.action_space,
                                                                               sub_id,
                                                                               add_alone_line=False))
                       for sub_id in range(type(tmp.env).n_sub)]
        tmp.tearDown()
        assert res == [0] * 14, f"found: {res}"
        assert res_noalone == [0] * 14, f"found: {res_noalone}"
        
        class SubMe2(TestActionSpace):
            def get_nb_bus(self):
                return 1
            def get_env_nm(self):
                return "l2rpn_idf_2023"
            
        tmp2 = SubMe2()
        tmp2.setUp()
        res = [len(tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,
                                                                        sub_id))
               for sub_id in range(type(tmp2.env).n_sub)]
        res_noalone = [len(tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,
                                                                                sub_id,
                                                                                add_alone_line=False))
                       for sub_id in range(type(tmp2.env).n_sub)]
        tmp2.tearDown()
        assert res == [0] * 118, f"found: {res}"
        assert res_noalone == [0] * 118, f"found: {res_noalone}"

    def test_3_busbars(self):
        """test :func:`grid2op.Action.SerializableActionSpace.get_all_unitary_topologies_set` 
        when there are 3 busbars per substation"""
        res = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space,
                                                                   _count_only=True)
        res_noalone  = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space,
                                                                            add_alone_line=False,
                                                                            _count_only=True)
        assert res == [3, 83, 5, 106, 33, 599, 5, 0, 33, 3, 3, 3, 10, 3], f"found: {res}"
        assert res_noalone == [0, 37, 3, 41, 11, 409, 0, 0, 11, 0, 0, 0, 4, 0], f"found: {res_noalone}"
        class SubMe2(TestActionSpace):
            def get_nb_bus(self):
                return 3
            def get_env_nm(self):
                return "l2rpn_idf_2023"
        tmp2 = SubMe2()
        tmp2.setUp() 
        th_vals = [0, 0, 4, 7, 11, 0, 0, 10, 0, 0, 125, 2108, 0, 0, 1711, 0, 162, 3, 37, 0, 0, 0, 37, 
                   4, 4, 0, 125, 0, 0, 4, 4, 41, 0, 37, 0, 0, 41, 0, 0, 37, 0, 409, 0, 0, 10, 10, 4, 0]
        for sub_id, th_val in zip(list(range(48)), th_vals):
            res_noalone  = tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,
                                                                                sub_id=sub_id,
                                                                                add_alone_line=False,
                                                                                _count_only=True)
            assert res_noalone[0] == th_val, f"error for sub_id {sub_id}: {res_noalone} vs {th_val}"
        
        if HAS_TIME_AND_MEMORY:            
            # takes 850s (13 minutes)
            res_noalone  = tmp2.env.action_space.get_all_unitary_topologies_set(tmp2.env.action_space,
                                                                                sub_id=48,
                                                                                add_alone_line=False,
                                                                                _count_only=True)
            assert res_noalone == 20698545, f"error for sub_id {48}: {res_noalone}"
        tmp2.tearDown() 
            
    def test_legacy_all_unitary_line_set_behaviour(self):
        """make sure nothing broke for 2 busbars per substation even if the implementation changes"""
        class SubMe(TestActionSpace):
            def get_nb_bus(self):
                return 2
            
        tmp = SubMe()
        tmp.setUp()
        res = len(tmp.env.action_space.get_all_unitary_line_set(tmp.env.action_space))
        res_simple = len(tmp.env.action_space.get_all_unitary_line_set_simple(tmp.env.action_space))
        tmp.tearDown()
        assert res == 5 * 20, f"found: {res}"
        assert res_simple == 2 * 20, f"found: {res_simple}"
        
        class SubMe2(TestActionSpace):
            def get_nb_bus(self):
                return 2
            def get_env_nm(self):
                return "l2rpn_idf_2023"
            
        tmp2 = SubMe2()
        tmp2.setUp()
        res = len(tmp2.env.action_space.get_all_unitary_line_set(tmp2.env.action_space))
        res_simple = len(tmp2.env.action_space.get_all_unitary_line_set_simple(tmp2.env.action_space))
        tmp2.tearDown()
        assert res == 5 * 186, f"found: {res}"
        assert res_simple == 2 * 186, f"found: {res_simple}"
        
    def test_get_all_unitary_line_set(self):
        """test the :func:`grid2op.Action.SerializableActionSpace.get_all_unitary_line_set` when 3 busbars"""
        res = len(self.env.action_space.get_all_unitary_line_set(self.env.action_space))
        assert res == (1 + 3*3) * 20, f"found: {res}"
        res = len(self.env.action_space.get_all_unitary_line_set_simple(self.env.action_space))
        assert res == 2 * 20, f"found: {res}"
        class SubMe2(TestActionSpace):
            def get_nb_bus(self):
                return 3
            def get_env_nm(self):
                return "l2rpn_idf_2023"
            
        tmp2 = SubMe2()
        tmp2.setUp()
        res = len(tmp2.env.action_space.get_all_unitary_line_set(tmp2.env.action_space))
        res_simple = len(tmp2.env.action_space.get_all_unitary_line_set_simple(tmp2.env.action_space))
        tmp2.tearDown()
        assert res == (1 + 3*3) * 186, f"found: {res}"
        assert res_simple == 2 * 186, f"found: {res_simple}"
               
               
class TestBackendAction(unittest.TestCase):
    pass


class TestPandapowerBackend(unittest.TestCase):
    pass


class TestObservation(unittest.TestCase):
    def test_action_space_get_back_to_ref_state(self):
        """test the :func:`grid2op.Action.SerializableActionSpace.get_back_to_ref_state` 
        when 3 busbars which could not be tested without observation"""
        pass

if __name__ == "__main__":
    unittest.main()
        