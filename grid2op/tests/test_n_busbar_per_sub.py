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


# test on a big computer only with lots of RAM, and lots of time available...
HAS_TIME_AND_MEMORY = False


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
        assert res == (1 + 3 * 3) * 186, f"found: {res}"
        assert res_simple == 2 * 186, f"found: {res_simple}"
               
               
class TestBackendAction(unittest.TestCase):
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

    def test_correct_last_topo(self):
        line_id = 0
        id_topo_or = type(self.env).line_or_pos_topo_vect[line_id]
        id_topo_ex = type(self.env).line_ex_pos_topo_vect[line_id]
        
        backend_action = self.env._backend_action
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == -1, f"{backend_action.current_topo.values[id_topo_or]} vs -1"
        assert backend_action.current_topo.values[id_topo_ex] == -1, f"{backend_action.current_topo.values[id_topo_ex]} vs -1"
        assert backend_action.last_topo_registered.values[id_topo_or] == 1, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 1"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 1, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 1"
        
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 2)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == 2, f"{backend_action.current_topo.values[id_topo_or]} vs 2"
        assert backend_action.current_topo.values[id_topo_ex] == 1, f"{backend_action.current_topo.values[id_topo_ex]} vs 1"
        assert backend_action.last_topo_registered.values[id_topo_or] == 2, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 2"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 1, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 1"
        
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == -1, f"{backend_action.current_topo.values[id_topo_or]} vs -1"
        assert backend_action.current_topo.values[id_topo_ex] == -1, f"{backend_action.current_topo.values[id_topo_ex]} vs -1"
        assert backend_action.last_topo_registered.values[id_topo_or] == 2, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 2"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 1, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 1"
        
        act = self.env.action_space({"set_bus": {"lines_ex_id": [(line_id, 3)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == 2, f"{backend_action.current_topo.values[id_topo_or]} vs 2"
        assert backend_action.current_topo.values[id_topo_ex] == 3, f"{backend_action.current_topo.values[id_topo_ex]} vs 3"
        assert backend_action.last_topo_registered.values[id_topo_or] == 2, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 2"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 3, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 3"    
         
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == -1, f"{backend_action.current_topo.values[id_topo_or]} vs -1"
        assert backend_action.current_topo.values[id_topo_ex] == -1, f"{backend_action.current_topo.values[id_topo_ex]} vs -1"
        assert backend_action.last_topo_registered.values[id_topo_or] == 2, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 2"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 3, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 3"   
        
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == -1, f"{backend_action.current_topo.values[id_topo_or]} vs -1"
        assert backend_action.current_topo.values[id_topo_ex] == -1, f"{backend_action.current_topo.values[id_topo_ex]} vs -1"
        assert backend_action.last_topo_registered.values[id_topo_or] == 2, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 2"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 3, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 3"   

        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 1)]}})
        backend_action += act
        backend_action.reset()
        assert backend_action.current_topo.values[id_topo_or] == 1, f"{backend_action.current_topo.values[id_topo_or]} vs 1"
        assert backend_action.current_topo.values[id_topo_ex] == 3, f"{backend_action.current_topo.values[id_topo_ex]} vs 3"
        assert backend_action.last_topo_registered.values[id_topo_or] == 1, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 1"
        assert backend_action.last_topo_registered.values[id_topo_ex] == 3, f"{backend_action.last_topo_registered.values[id_topo_or]} vs 3"   

    def test_call(self):
        cls = type(self.env)
        line_id = 0
        id_topo_or = cls.line_or_pos_topo_vect[line_id]
        id_topo_ex = cls.line_ex_pos_topo_vect[line_id]
        
        backend_action = self.env._backend_action
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == -1
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == -1
        backend_action.reset()
        
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 2)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == 2
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == 1
        backend_action.reset()
        
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == -1
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == -1
        backend_action.reset()
        
        act = self.env.action_space({"set_bus": {"lines_ex_id": [(line_id, 3)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == 2
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == 3
        backend_action.reset() 
         
        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, -1)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == -1
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == -1
        backend_action.reset() 

        act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, 1)]}})
        backend_action += act
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            topo__,
            shunts__,
        ) = backend_action()
        assert topo__.values[cls.line_or_pos_topo_vect[line_id]] == 1
        assert topo__.values[cls.line_ex_pos_topo_vect[line_id]] == 3
        backend_action.reset() 
        
        
class TestPandapowerBackend_3busbars(unittest.TestCase):
    def get_nb_bus(self):
        return 3
    
    def get_env_nm(self):
        return "educ_case14_storage"
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.get_env_nm(),
                                    backend=PandaPowerBackend(),
                                    action_class=CompleteAction,
                                    test=True,
                                    n_busbar=self.get_nb_bus(),
                                    _add_to_name=type(self).__name__ + f'_{self.get_nb_bus()}')
        self.list_loc_bus = [-1] + list(range(1, type(self.env).n_busbar_per_sub + 1))
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_right_bus_made(self):
        assert self.env.backend._grid.bus.shape[0] == self.get_nb_bus() * type(self.env).n_sub
        assert (~self.env.backend._grid.bus.iloc[type(self.env).n_sub:]["in_service"]).all()

    @staticmethod
    def _aux_find_sub(env, obj_col):
        """find a sub with 4 elements, the type of elements and at least 2 lines"""
        cls = type(env)
        res = None
        for sub_id in range(cls.n_sub):
            this_sub_mask = cls.grid_objects_types[:,cls.SUB_COL] == sub_id
            this_sub = cls.grid_objects_types[this_sub_mask, :]
            if this_sub.shape[0] <= 3:
                # not enough element
                continue
            if (this_sub[:, obj_col] == -1).all():
                # no load
                continue
            if ((this_sub[:, cls.LOR_COL] != -1) | (this_sub[:, cls.LEX_COL] != -1)).sum() <= 1:
                # only 1 line
                continue
            el_id = this_sub[this_sub[:, obj_col] != -1, obj_col][0]
            if (this_sub[:, cls.LOR_COL] != -1).any():
                line_or_id = this_sub[this_sub[:, cls.LOR_COL] != -1, cls.LOR_COL][0]
                line_ex_id = None
            else:
                line_or_id = None
                line_ex_id = this_sub[this_sub[:, cls.LEX_COL] != -1, cls.LEX_COL][0]
            res = (sub_id, el_id, line_or_id, line_ex_id)
            break
        return res
    
    @staticmethod
    def _aux_find_sub_shunt(env):
        """find a sub with 4 elements, the type of elements and at least 2 lines"""
        cls = type(env)
        res = None
        for el_id in range(cls.n_shunt):
            sub_id = cls.shunt_to_subid[el_id]
            this_sub_mask = cls.grid_objects_types[:,cls.SUB_COL] == sub_id
            this_sub = cls.grid_objects_types[this_sub_mask, :]
            if this_sub.shape[0] <= 3:
                # not enough element
                continue
            if ((this_sub[:, cls.LOR_COL] != -1) | (this_sub[:, cls.LEX_COL] != -1)).sum() <= 1:
                # only 1 line
                continue
            if (this_sub[:, cls.LOR_COL] != -1).any():
                line_or_id = this_sub[this_sub[:, cls.LOR_COL] != -1, cls.LOR_COL][0]
                line_ex_id = None
            else:
                line_or_id = None
                line_ex_id = this_sub[this_sub[:, cls.LEX_COL] != -1, cls.LEX_COL][0]
            res = (sub_id, el_id, line_or_id, line_ex_id)
            break
        return res
        
    def test_move_load(self):
        cls = type(self.env)            
        res = self._aux_find_sub(self.env, cls.LOA_COL)
        if res is None:
            raise RuntimeError(f"Cannot carry the test 'test_move_load' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if line_or_id is not None:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = sub_id + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.load.iloc[el_id]["bus"] == global_bus
                if line_or_id is not None:
                    assert self.env.backend._grid.line.iloc[line_or_id]["from_bus"] == global_bus
                else:
                    assert self.env.backend._grid.line.iloc[line_ex_id]["to_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.load.iloc[el_id]["in_service"]
                if line_or_id is not None:
                    assert not self.env.backend._grid.line.iloc[line_or_id]["in_service"]
                else:
                    assert not self.env.backend._grid.line.iloc[line_ex_id]["in_service"]
            topo_vect = self.env.backend._get_topo_vect()
            assert topo_vect[cls.load_pos_topo_vect[el_id]] == new_bus, f"{topo_vect[cls.load_pos_topo_vect[el_id]]} vs {new_bus}"
        
    def test_move_gen(self):
        cls = type(self.env)            
        res = self._aux_find_sub(self.env, cls.GEN_COL)
        if res is None:
            raise RuntimeError(f"Cannot carry the test 'test_move_gen' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if line_or_id is not None:
                act = self.env.action_space({"set_bus": {"generators_id": [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"set_bus": {"generators_id": [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = sub_id + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.gen.iloc[el_id]["bus"] == global_bus
                if line_or_id is not None:
                    assert self.env.backend._grid.line.iloc[line_or_id]["from_bus"] == global_bus
                else:
                    assert self.env.backend._grid.line.iloc[line_ex_id]["to_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.gen.iloc[el_id]["in_service"]
                if line_or_id is not None:
                    assert not self.env.backend._grid.line.iloc[line_or_id]["in_service"]
                else:
                    assert not self.env.backend._grid.line.iloc[line_ex_id]["in_service"]
            topo_vect = self.env.backend._get_topo_vect()
            assert topo_vect[cls.gen_pos_topo_vect[el_id]] == new_bus, f"{topo_vect[cls.gen_pos_topo_vect[el_id]]} vs {new_bus}"
        
    def test_move_storage(self):
        cls = type(self.env)            
        res = self._aux_find_sub(self.env, cls.STORAGE_COL)
        if res is None:
            raise RuntimeError(f"Cannot carry the test 'test_move_storage' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if line_or_id is not None:
                act = self.env.action_space({"set_bus": {"storages_id": [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"set_bus": {"storages_id": [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = sub_id + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.storage.iloc[el_id]["bus"] == global_bus
                assert self.env.backend._grid.storage.iloc[el_id]["in_service"], f"storage should not be deactivated"
                if line_or_id is not None:
                    assert self.env.backend._grid.line.iloc[line_or_id]["from_bus"] == global_bus
                else:
                    assert self.env.backend._grid.line.iloc[line_ex_id]["to_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.storage.iloc[el_id]["in_service"], f"storage should be deactivated"
                if line_or_id is not None:
                    assert not self.env.backend._grid.line.iloc[line_or_id]["in_service"]
                else:
                    assert not self.env.backend._grid.line.iloc[line_ex_id]["in_service"]
            topo_vect = self.env.backend._get_topo_vect()
            assert topo_vect[cls.storage_pos_topo_vect[el_id]] == new_bus, f"{topo_vect[cls.storage_pos_topo_vect[el_id]]} vs {new_bus}"
    
    def test_move_line_or(self):
        cls = type(self.env)            
        line_id = 0
        for new_bus in self.list_loc_bus:
            act = self.env.action_space({"set_bus": {"lines_or_id": [(line_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = cls.line_or_to_subid[line_id] + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.line.iloc[line_id]["from_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.line.iloc[line_id]["in_service"]
            self.env.backend.line_status[:] = self.env.backend._get_line_status()  # otherwise it's not updated
            topo_vect = self.env.backend._get_topo_vect()
            assert topo_vect[cls.line_or_pos_topo_vect[line_id]] == new_bus, f"{topo_vect[cls.line_or_pos_topo_vect[line_id]]} vs {new_bus}"
                
    def test_move_line_ex(self):
        cls = type(self.env)            
        line_id = 0
        for new_bus in self.list_loc_bus:
            act = self.env.action_space({"set_bus": {"lines_ex_id": [(line_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = cls.line_ex_to_subid[line_id] + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.line.iloc[line_id]["to_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.line.iloc[line_id]["in_service"]
            self.env.backend.line_status[:] = self.env.backend._get_line_status()  # otherwise it's not updated
            topo_vect = self.env.backend._get_topo_vect()
            assert topo_vect[cls.line_ex_pos_topo_vect[line_id]] == new_bus, f"{topo_vect[cls.line_ex_pos_topo_vect[line_id]]} vs {new_bus}"
            
    def test_move_shunt(self):
        cls = type(self.env)            
        res = self._aux_find_sub_shunt(self.env)
        if res is None:
            raise RuntimeError(f"Cannot carry the test 'test_move_load' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if line_or_id is not None:
                act = self.env.action_space({"shunt": {"set_bus": [(el_id, new_bus)]}, "set_bus": {"lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"shunt": {"set_bus": [(el_id, new_bus)]}, "set_bus": {"lines_ex_id": [(line_ex_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            global_bus = sub_id + (new_bus -1) * cls.n_sub 
            if new_bus >= 1:
                assert self.env.backend._grid.shunt.iloc[el_id]["bus"] == global_bus
                if line_or_id is not None:
                    assert self.env.backend._grid.line.iloc[line_or_id]["from_bus"] == global_bus
                else:
                    assert self.env.backend._grid.line.iloc[line_ex_id]["to_bus"] == global_bus
                assert self.env.backend._grid.bus.loc[global_bus]["in_service"]
            else:
                assert not self.env.backend._grid.shunt.iloc[el_id]["in_service"]
                if line_or_id is not None:
                    assert not self.env.backend._grid.line.iloc[line_or_id]["in_service"]
                else:
                    assert not self.env.backend._grid.line.iloc[line_ex_id]["in_service"]
    
    def test_check_kirchoff(self):
        cls = type(self.env)            
        res = self._aux_find_sub(self.env, cls.LOA_COL)
        if res is None:
            raise RuntimeError("Cannot carry the test 'test_move_load' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if new_bus <= -1:
                continue
            if line_or_id is not None:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
            bk_act = self.env._backend_action_class()
            bk_act += act
            self.env.backend.apply_action(bk_act)
            conv, maybe_exc = self.env.backend.runpf()
            assert conv, f"error : {maybe_exc}"
            p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
            # assert laws are met
            assert np.abs(p_subs).max() <= 1e-5, f"error for busbar {new_bus}: {np.abs(p_subs).max():.2e}"
            assert np.abs(q_subs).max() <= 1e-5, f"error for busbar {new_bus}: {np.abs(q_subs).max():.2e}"
            assert np.abs(p_bus).max() <= 1e-5, f"error for busbar {new_bus}: {np.abs(p_bus).max():.2e}"
            assert np.abs(q_bus).max() <= 1e-5, f"error for busbar {new_bus}: {np.abs(q_bus).max():.2e}"
            assert np.abs(diff_v_bus).max() <= 1e-5, f"error for busbar {new_bus}: {np.abs(diff_v_bus).max():.2e}"


class TestPandapowerBackend_1busbar(TestPandapowerBackend_3busbars):
    def get_nb_bus(self):
        return 1
        
        
class TestObservation_3busbars(unittest.TestCase):
    def get_nb_bus(self):
        return 3
    
    def get_env_nm(self):
        return "educ_case14_storage"
    
    def get_reset_kwargs(self) -> dict:
        return dict(seed=0, options={"time serie id": 0})
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.get_env_nm(),
                                    backend=PandaPowerBackend(),
                                    action_class=CompleteAction,
                                    test=True,
                                    n_busbar=self.get_nb_bus(),
                                    _add_to_name=type(self).__name__ + f'_{self.get_nb_bus()}')
        param = self.env.parameters
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.MAX_LINE_STATUS_CHANGED = 99999
        param.MAX_SUB_CHANGED = 99999
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        self.env.reset(**self.get_reset_kwargs())
        self.list_loc_bus = list(range(1, type(self.env).n_busbar_per_sub + 1))
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_get_simulator(self):
        obs = self.env.reset(**self.get_reset_kwargs())
        sim = obs.get_simulator()
        assert type(sim.backend).n_busbar_per_sub == self.get_nb_bus()
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        if res is None:
            raise RuntimeError(f"Cannot carry the test 'test_get_simulator' as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if line_or_id is not None:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
            else:
                act = self.env.action_space({"set_bus": {"loads_id": [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
            sim2 = sim.predict(act)
            global_bus = sub_id + (new_bus -1) * type(self.env).n_sub 
            assert sim2.backend._grid.load["bus"].iloc[el_id] == global_bus
            
    def _aux_build_act(self, res, new_bus, el_keys):
        """res: output of TestPandapowerBackend_3busbars._aux_find_sub"""
        if res is None:
            raise RuntimeError(f"Cannot carry the test as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        if line_or_id is not None:
            act = self.env.action_space({"set_bus": {el_keys: [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
        else:
            act = self.env.action_space({"set_bus": {el_keys: [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
        return act
    
    @staticmethod        
    def _aux_aux_build_act(env, res, new_bus, el_keys):
        """res: output of TestPandapowerBackend_3busbars._aux_find_sub"""
        if res is None:
            raise RuntimeError(f"Cannot carry the test as "
                               "there are no suitable subastation in your grid.")
        (sub_id, el_id, line_or_id, line_ex_id) = res
        if line_or_id is not None:
            act = env.action_space({"set_bus": {el_keys: [(el_id, new_bus)], "lines_or_id": [(line_or_id, new_bus)]}})
        else:
            act = env.action_space({"set_bus": {el_keys: [(el_id, new_bus)], "lines_ex_id": [(line_ex_id, new_bus)]}})
        return act
        
    def test_get_forecasted_env(self):
        obs = self.env.reset(**self.get_reset_kwargs())
        for_env = obs.get_forecast_env()
        assert type(for_env).n_busbar_per_sub == self.get_nb_bus()
        for_obs = for_env.reset()
        assert type(for_obs).n_busbar_per_sub == self.get_nb_bus()
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            for_env = obs.get_forecast_env()
            act = self._aux_build_act(res, new_bus, "loads_id")
            sim_obs, sim_r, sim_d, sim_info = for_env.step(act)
            assert not sim_d, f"{sim_info['exception']}"
            assert sim_obs.load_bus[el_id] == new_bus, f"{sim_obs.load_bus[el_id]} vs {new_bus}"
    
    def test_add(self):
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs_pus_act = obs + act
            assert obs_pus_act.load_bus[el_id] == new_bus, f"{obs_pus_act.load_bus[el_id]} vs {new_bus}"
            
    def test_simulate(self):
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            sim_obs, sim_r, sim_d, sim_info = obs.simulate(act)
            assert not sim_d, f"{sim_info['exception']}"
            assert sim_obs.load_bus[el_id] == new_bus, f"{sim_obs.load_bus[el_id]} vs {new_bus}"
    
    def test_action_space_get_back_to_ref_state(self):
        """test the :func:`grid2op.Action.SerializableActionSpace.get_back_to_ref_state` 
        when 3 busbars which could not be tested without observation"""
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            if new_bus == 1:
                # nothing to do if everything is moved to bus 1
                continue
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            acts = self.env.action_space.get_back_to_ref_state(obs)
            assert "substation" in acts
            assert len(acts["substation"]) == 1
            act_to_ref = acts["substation"][0]
            assert act_to_ref.load_set_bus[el_id] == 1
            if line_or_id is not None:
                assert act_to_ref.line_or_set_bus[line_or_id] == 1
            if line_ex_id is not None:
                assert act_to_ref.line_ex_set_bus[line_ex_id] == 1
    
    def test_connectivity_matrix(self):
        cls = type(self.env)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["exception"], "there should not have any exception (action should be legal)"
            conn_mat = obs.connectivity_matrix()
            assert conn_mat.shape == (cls.dim_topo, cls.dim_topo)
            if new_bus == 1:
                min_sub = np.sum(cls.sub_info[:sub_id])
                max_sub = min_sub + cls.sub_info[sub_id]
                assert (conn_mat[min_sub:max_sub, min_sub:max_sub] == 1.).all()
            else:
                el_topov = cls.load_pos_topo_vect[el_id]
                line_pos_topov = cls.line_or_pos_topo_vect[line_or_id] if line_or_id is not None else cls.line_ex_pos_topo_vect[line_ex_id]
                line_pos_topo_other = cls.line_ex_pos_topo_vect[line_or_id] if line_or_id is not None else cls.line_or_pos_topo_vect[line_ex_id]
                assert conn_mat[el_topov, line_pos_topov] == 1.
                assert conn_mat[line_pos_topov, el_topov] == 1.
                for el in range(cls.dim_topo):
                    if el == line_pos_topov:
                        continue
                    if el == el_topov:
                        continue
                    if el == line_pos_topo_other:
                        # other side of the line is connected to it
                        continue
                    assert conn_mat[el_topov, el] == 0., f"error for {new_bus}: ({el_topov}, {el}) appears to be connected: {conn_mat[el_topov, el]}"
                    assert conn_mat[el, el_topov] == 0., f"error for {new_bus}: ({el}, {el_topov}) appears to be connected: {conn_mat[el, el_topov]}"
                    assert conn_mat[line_pos_topov, el] == 0., f"error for {new_bus}: ({line_pos_topov}, {el}) appears to be connected: {conn_mat[line_pos_topov, el]}"
                    assert conn_mat[el, line_pos_topov] == 0., f"error for {new_bus}: ({el}, {line_pos_topov}) appears to be connected: {conn_mat[el, line_pos_topov]}"
    
    def test_bus_connectivity_matrix(self):
        cls = type(self.env)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["exception"], "there should not have any exception (action should be legal)"
            conn_mat, (lor_ind, lex_ind) = obs.bus_connectivity_matrix(return_lines_index=True)
            if new_bus == 1:
                assert conn_mat.shape == (cls.n_sub, cls.n_sub)
            else:
                assert conn_mat.shape == (cls.n_sub + 1, cls.n_sub + 1)
                new_bus_id = lor_ind[line_or_id] if line_or_id else lex_ind[line_ex_id]
                bus_other = lex_ind[line_or_id] if line_or_id else lor_ind[line_ex_id]
                assert conn_mat[new_bus_id, bus_other] == 1.
                assert conn_mat[bus_other, new_bus_id] == 1.
                assert conn_mat[new_bus_id, sub_id] == 0.
                assert conn_mat[sub_id, new_bus_id] == 0.
                
    def test_flow_bus_matrix(self):
        cls = type(self.env)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["exception"], "there should not have any exception (action should be legal)"
            conn_mat, (load_bus, prod_bus, stor_bus, lor_ind, lex_ind) = obs.flow_bus_matrix()
            if new_bus == 1:
                assert conn_mat.shape == (cls.n_sub, cls.n_sub)
            else:
                assert conn_mat.shape == (cls.n_sub + 1, cls.n_sub + 1)
                new_bus_id = lor_ind[line_or_id] if line_or_id else lex_ind[line_ex_id]
                bus_other = lex_ind[line_or_id] if line_or_id else lor_ind[line_ex_id]
                assert conn_mat[new_bus_id, bus_other] != 0. # there are some flows from these 2 buses
                assert conn_mat[bus_other, new_bus_id] != 0. # there are some flows from these 2 buses
                assert conn_mat[new_bus_id, sub_id] == 0.
                assert conn_mat[sub_id, new_bus_id] == 0.
    
    def test_get_energy_graph(self):
        cls = type(self.env)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["exception"], "there should not have any exception (action should be legal)"
            graph = obs.get_energy_graph()
            if new_bus == 1:
                assert len(graph.nodes) == cls.n_sub
                continue
            # if I end up here it's because new_bus >= 2
            assert len(graph.nodes) == cls.n_sub + 1
            new_bus_id = cls.n_sub  # this bus has been added
            bus_other = cls.line_ex_to_subid[line_or_id] if line_or_id else cls.line_or_to_subid[line_ex_id]
            assert (new_bus_id, bus_other) in graph.edges
            edge = graph.edges[(new_bus_id, bus_other)]
            node = graph.nodes[new_bus_id]
            assert node["local_bus_id"] == new_bus
            assert node["global_bus_id"] == sub_id + (new_bus - 1) * cls.n_sub
            if line_or_id is not None:
                assert edge["bus_or"] == new_bus
                assert edge["global_bus_or"] == sub_id + (new_bus - 1) * cls.n_sub
            else:
                assert edge["bus_ex"] == new_bus
                assert edge["global_bus_ex"] == sub_id + (new_bus - 1) * cls.n_sub
                
    def test_get_elements_graph(self):
        cls = type(self.env)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        for new_bus in self.list_loc_bus:
            act = self._aux_build_act(res, new_bus, "loads_id")
            obs, reward, done, info = self.env.step(act)
            assert not done
            assert not info["exception"], "there should not have any exception (action should be legal)"
            graph = obs.get_elements_graph()
            global_bus_id = sub_id + (new_bus - 1) * cls.n_sub
            node_bus_id = graph.graph['bus_nodes_id'][global_bus_id]
            node_load_id = graph.graph['load_nodes_id'][el_id]
            node_line_id = graph.graph['line_nodes_id'][line_or_id] if line_or_id is not None else graph.graph['line_nodes_id'][line_ex_id]
            node_load = graph.nodes[node_load_id]
            node_line = graph.nodes[node_line_id]     
            assert len(graph.graph["bus_nodes_id"]) == cls.n_busbar_per_sub * cls.n_sub       
            
            # check the bus
            for node_id in graph.graph["bus_nodes_id"]:
                assert "global_id" in graph.nodes[node_id], "key 'global_id' should be in the node"
            if new_bus == 1:
                for node_id in graph.graph["bus_nodes_id"][cls.n_sub:]:
                    assert not graph.nodes[node_id]["connected"], f"bus (global id {graph.nodes[node_id]['global_id']}) represented by node {node_id} should not be connected"
            else:
                for node_id in graph.graph["bus_nodes_id"][cls.n_sub:]:
                    if graph.nodes[node_id]['global_id'] != global_bus_id:
                        assert not graph.nodes[node_id]["connected"], f"bus (global id {graph.nodes[node_id]['global_id']}) represented by node {node_id} should not be connected"
                    else:
                        assert graph.nodes[node_id]["connected"], f"bus (global id {graph.nodes[node_id]['global_id']}) represented by node {node_id} should be connected"
                        
            # check the load
            edge_load_id = node_load["bus_node_id"]
            assert node_load["local_bus"] == new_bus
            assert node_load["global_bus"] == global_bus_id
            assert (node_load_id, edge_load_id) in graph.edges
            
            # check lines
            side = "or" if line_or_id is not None else "ex"
            edge_line_id = node_line[f"bus_node_id_{side}"]
            assert node_line[f"local_bus_{side}"] == new_bus
            assert node_line[f"global_bus_{side}"] == global_bus_id
            assert (node_line_id, edge_line_id) in graph.edges
   
   
class TestObservation_1busbar(TestObservation_3busbars):
    def get_nb_bus(self):
        return 1               
                
                
class TestEnv(unittest.TestCase):
    def get_nb_bus(self):
        return 3
    
    def get_env_nm(self):
        return "educ_case14_storage"
    
    def get_reset_kwargs(self) -> dict:
        return dict(seed=0, options={"time serie id": 0})
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.get_env_nm(),
                                    backend=PandaPowerBackend(),
                                    action_class=CompleteAction,
                                    test=True,
                                    n_busbar=self.get_nb_bus(),
                                    _add_to_name=type(self).__name__ + f'_{self.get_nb_bus()}')
        param = self.env.parameters
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.MAX_LINE_STATUS_CHANGED = 99999
        param.MAX_SUB_CHANGED = 99999
        self.env.change_parameters(param)
        self.env.change_forecast_parameters(param)
        self.env.reset(**self.get_reset_kwargs())
        self.list_loc_bus = list(range(1, type(self.env).n_busbar_per_sub + 1))
        self.max_iter = 10
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_go_to_end(self):
        self.env.set_max_iter(self.max_iter)
        obs = self.env.reset(**self.get_reset_kwargs())
        i = 0
        done = False
        while not done:
            obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == 10, f"{i} instead of 10"
        
    def test_can_put_on_3(self):
        self.env.set_max_iter(self.max_iter)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        act = TestObservation_3busbars._aux_aux_build_act(self.env, res, self.get_nb_bus(), "loads_id")
        i = 0
        done = False
        while not done:
            if i == 0:
                obs, reward, done, info = self.env.step(act)
            else:
                obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == 10, f"{i} instead of 10"
        
    def test_can_move_from_3(self):
        self.env.set_max_iter(self.max_iter)
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        act = TestObservation_3busbars._aux_aux_build_act(self.env, res, self.get_nb_bus(), "loads_id")
        i = 0
        done = False
        while not done:
            if i == 0:
                # do the action to set on a busbar 3
                obs, reward, done, info = self.env.step(act)
                assert not done
                assert not info["exception"]
            elif i == 1:
                # do the opposite action
                dict_act = obs.get_back_to_ref_state()
                assert "substation" in dict_act
                li_act = dict_act["substation"]
                assert len(li_act) == 1
                act = li_act[0]
                obs, reward, done, info = self.env.step(act)
                assert not done
                assert not info["exception"]
            else:
                obs, reward, done, info = self.env.step(self.env.action_space())
            i += 1
        assert i == 10, f"{i} instead of 10"
        
    def _aux_alone_done(self, key="loads_id"):
        if self.get_nb_bus() <= 2:
            self.skipTest("Need at leat two busbars")
        obs = self.env.reset(**self.get_reset_kwargs())
        act = self.env.action_space({"set_bus": {key: [(0, self.get_nb_bus())]}})
        obs, reward, done, info = self.env.step(act)
        assert done
    
    def test_load_alone_done(self):
        self._aux_alone_done("loads_id")
        
    def test_gen_alone_done(self):
        self._aux_alone_done("generators_id")
        
    def test_simulate(self):
        """test the obs.simulate(...) works with different number of busbars"""
        obs = self.env.reset(**self.get_reset_kwargs())
        res = TestPandapowerBackend_3busbars._aux_find_sub(self.env, type(self.env).LOA_COL)
        (sub_id, el_id, line_or_id, line_ex_id) = res
        act = TestObservation_3busbars._aux_aux_build_act(self.env, res, self.get_nb_bus(), "loads_id")
        sim_obs, sim_r, sim_d, sim_i = obs.simulate(act)
        assert not sim_d
        assert not sim_i["exception"]
        
    
class TestGym(unittest.TestCase):
    pass


class TestRules(unittest.TestCase):
    """test the rules for the reco / deco of line works also when >= 3 busbars, 
    also ttests the act.get_impact()...
    """
    pass


if __name__ == "__main__":
    unittest.main()
        