# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import multiprocessing as mp
import sys
from typing import Optional
import warnings
import unittest
import importlib
import numpy as np
from gymnasium.vector import AsyncVectorEnv

os.environ["grid2op_class_in_file"] = "true"

import grid2op
from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
from grid2op.Runner import Runner
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Action.actionSpace import ActionSpace
from grid2op.Environment import (Environment,
                                 MaskedEnvironment,
                                 TimedOutEnvironment,
                                 SingleEnvMultiProcess,
                                 MultiMixEnvironment)
from grid2op.Exceptions import NoForecastAvailable
from grid2op.gym_compat import (GymEnv,
                                BoxGymActSpace,
                                BoxGymObsSpace,
                                DiscreteActSpace,
                                MultiDiscreteActSpace)

# TODO feature: in the make add a kwargs to deactivate this

# TODO test the runner saved classes and reload

# TODO two envs same name => now diff classes

# TODO test add_to_name
# TODO test noshunt
# TODO grid2op compat version

# TODO test backend converters
# TODO mode to propagate the "pointer" (this_local_dir = tempfile.TemporaryDirectory(dir=sys_path))
#      in all copy of the environment instead of keeping it only for the first one
# TODO test all type of backend in the observation space, including the deactivate forecast, reactivate forecast, the different backend etc.

class _ThisAgentTest(BaseAgent):
    def __init__(self,
                 action_space: ActionSpace,
                 _read_from_local_dir,
                 _name_cls_obs,
                 _name_cls_act,
                 ):
        super().__init__(action_space)
        self._read_from_local_dir = _read_from_local_dir
        self._name_cls_obs = _name_cls_obs
        self._name_cls_act = _name_cls_act
        
    def act(self, observation: BaseObservation, reward: float, done: bool = False) -> BaseAction:
        supermodule_nm, module_nm = os.path.split(self._read_from_local_dir)
        super_module = importlib.import_module(module_nm, supermodule_nm) 
        
        # check observation
        this_module = importlib.import_module(f"{module_nm}.{self._name_cls_obs}_file", super_module)       
        if hasattr(this_module, self._name_cls_obs):
            this_class_obs = getattr(this_module, self._name_cls_obs)
        else:
            raise RuntimeError(f"class {self._name_cls_obs} not found")
        assert isinstance(observation, this_class_obs)
        
        # check action
        this_module = importlib.import_module(f"{module_nm}.{self._name_cls_act}_file", super_module)       
        if hasattr(this_module, self._name_cls_act):
            this_class_act = getattr(this_module, self._name_cls_act)
        else:
            raise RuntimeError(f"class {self._name_cls_act} not found")
        res = super().act(observation, reward, done)
        assert isinstance(res, this_class_act)
        return res


class AutoClassMakeTester(unittest.TestCase):
    """test that the kwargs `class_in_file=False` erase the default behaviour """
    def test_in_make(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, class_in_file=False)
        assert env._read_from_local_dir is None
        assert not env.classes_are_in_files()
    
    
class AutoClassInFileTester(unittest.TestCase):
    def get_env_name(self):
        return "l2rpn_case14_sandbox"
    
    def setUp(self) -> None:
        self.max_iter = 10
        return super().setUp()
    
    def _do_test_runner(self):
        # false for multi process env
        return True
    
    def _do_test_copy(self):
        # for for multi process env
        return True
    
    def _do_test_obs_env(self):
        return True
    
    def _aux_make_env(self, env: Optional[Environment]=None):
        assert USE_CLASS_IN_FILE
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(self.get_env_name(), test=True)
        assert env.classes_are_in_files()
        return env
    
    def _aux_get_obs_cls(self):
        return "CompleteObservation_{}"
    
    def _aux_get_act_cls(self):
        return "PlayableAction_{}"
    
    def test_all_classes_from_file(self,
                                   env: Optional[Environment]=None,
                                   classes_name=None,
                                   name_complete_obs_cls="CompleteObservation_{}",
                                   name_observation_cls=None,
                                   name_action_cls=None):
        if classes_name is None:
            classes_name = self.get_env_name()
        if name_observation_cls is None:
            name_observation_cls = self._aux_get_obs_cls().format(classes_name)
        if name_action_cls is None:
            name_action_cls = self._aux_get_act_cls().format(classes_name)
            
        name_action_cls = name_action_cls.format(classes_name)
        env = self._aux_make_env(env)
        names_cls = [f"ActionSpace_{classes_name}",
                     f"_BackendAction_{classes_name}",
                     f"CompleteAction_{classes_name}",
                     name_observation_cls.format(classes_name),
                     name_complete_obs_cls.format(classes_name),
                     f"DontAct_{classes_name}",
                     f"_ObsEnv_{classes_name}",
                     f"ObservationSpace_{classes_name}",
                     f"PandaPowerBackend_{classes_name}",
                     name_action_cls,
                     f"VoltageOnlyAction_{classes_name}"
                     ]
        names_attr = ["action_space",
                      "_backend_action_class",
                      "_complete_action_cls",
                      "_observationClass",
                      None,  # Complete Observation in the forecast !
                      None, # DONT ACT not int ENV directly
                      None, #   ObsEnv NOT IN ENV,
                      "observation_space",
                      "backend",
                      "_actionClass",
                      None, # VoltageOnlyAction not in env
                      ]
        
        # NB: these imports needs to be consistent with what is done in
        # base_env.generate_classes() and gridobj.init_grid(...)
        supermodule_nm, module_nm = os.path.split(env._read_from_local_dir)
        super_module = importlib.import_module(module_nm, supermodule_nm)
        for name_cls, name_attr in zip(names_cls, names_attr):
            this_module = importlib.import_module(f"{module_nm}.{name_cls}_file", super_module)
            if hasattr(this_module, name_cls):
                this_class = getattr(this_module, name_cls)
            else:
                raise RuntimeError(f"class {name_cls} not found")
            if name_attr is not None:
                the_attr = getattr(env, name_attr)
                if isinstance(the_attr, type):
                    assert the_attr is this_class, f"error for {the_attr} vs {this_class} env.{name_attr}"
                else:
                    assert type(the_attr) is this_class, f"error for {type(the_attr)} vs {this_class} (env.{name_attr})"
                assert this_class._CLS_DICT is not None, f'error for {name_cls}'
                assert this_class._CLS_DICT_EXTENDED is not None, f'error for {name_cls}'
                
            # additional check for some attributes
            if name_cls == f"ActionSpace_{classes_name}":
                assert type(env._helper_action_env) is this_class
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.observation_space.obs_env._helper_action_env) is this_class, f"{type(env.observation_space.obs_env._helper_action_env)}"
                if env._voltage_controler is not None:
                    # not in _ObsEnv
                    assert type(env._voltage_controler.action_space) is this_class
                if env.chronics_handler.action_space is not None:
                    # not in _ObsEnv
                    assert type(env.chronics_handler.action_space) is this_class
                    assert env.chronics_handler.action_space is env._helper_action_env
            elif name_cls == f"_BackendAction_{classes_name}":
                assert env.backend.my_bk_act_class is this_class
                assert isinstance(env._backend_action, this_class)
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert env.observation_space.obs_env._backend_action_class is this_class
                    assert env.observation_space.obs_env.backend.my_bk_act_class is this_class
                    assert isinstance(env.observation_space.obs_env._backend_action, this_class)
            elif name_cls == f"CompleteAction_{classes_name}":
                assert env.backend._complete_action_class is this_class
                
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert env.observation_space.obs_env._complete_action_cls is this_class
                    assert env.observation_space.obs_env.backend._complete_action_class is this_class
                
                    assert env.observation_space.obs_env._actionClass is this_class
                
                assert env._helper_action_env.subtype is this_class
            elif name_cls == name_observation_cls.format(classes_name):
                # observation of the env
                assert env._observation_space.subtype is this_class
                if env.current_obs is not None:
                    # not in _ObsEnv
                    assert isinstance(env.current_obs, this_class)
                if env._last_obs is not None:
                    # not in _ObsEnv
                    assert isinstance(env._last_obs, this_class)
            elif name_cls == name_observation_cls.format(classes_name):
                # observation of the forecast
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert env._observation_space.obs_env._observation_space.subtype is this_class
                    if env.observation_space.obs_env.current_obs is not None:
                        # not in _ObsEnv
                        assert isinstance(env.observation_space.obs_env.current_obs, this_class)
                    if env.observation_space.obs_env._last_obs is not None:
                        # not in _ObsEnv
                        assert isinstance(env.observation_space.obs_env._last_obs, this_class)
            elif name_cls == f"DontAct_{classes_name}":
                assert env._oppSpace.action_space.subtype is this_class
                assert env._opponent.action_space.subtype is this_class
            elif name_cls == f"_ObsEnv_{classes_name}":
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.observation_space.obs_env) is this_class
                    assert isinstance(env.observation_space.obs_env, this_class)
                if env.current_obs is not None and env.current_obs._obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.current_obs._obs_env) is this_class, f"{type(env.current_obs._obs_env)}"
                    assert isinstance(env.observation_space.obs_env, this_class)
                if env._last_obs is not None and env._last_obs._obs_env is not None:
                    # not in _ObsEnv
                    assert type(env._last_obs._obs_env) is this_class, f"{type(env._last_obs._obs_env)}"
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert env.current_obs._obs_env is env.observation_space.obs_env
                    assert env._last_obs._obs_env is env.observation_space.obs_env
            elif name_cls == f"ObservationSpace_{classes_name}":
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.observation_space.obs_env._observation_space) is this_class
                    assert type(env.observation_space.obs_env._ptr_orig_obs_space) is this_class, f"{type(env.observation_space.obs_env._ptr_orig_obs_space)}"
                    
                    assert env.observation_space.obs_env._ptr_orig_obs_space is env._observation_space, f"{type(env.observation_space.obs_env._ptr_orig_obs_space)}"
            elif name_cls == name_action_cls:
                assert env._action_space.subtype is this_class
                # assert env.observation_space.obs_env._actionClass is this_class  # not it's a complete action apparently
            elif name_cls == f"VoltageOnlyAction_{classes_name}":
                if env._voltage_controler is not None:
                    # not in _ObsEnv
                    assert env._voltage_controler.action_space.subtype is this_class
        # TODO test current_obs and _last_obs
        
    def test_all_classes_from_file_env_after_reset(self, env: Optional[Environment]=None):
        """test classes are still consistent even after a call to env.reset() and obs.simulate()"""
        env = self._aux_make_env(env)
        obs = env.reset()
        self.test_all_classes_from_file(env=env)
        try:
            obs.simulate(env.action_space())
            self.test_all_classes_from_file(env=env)
        except NoForecastAvailable:
            # cannot do this test if the "original" env is a _Forecast env:
            # for l2rpn_case14_sandbox only 1 step ahead forecast are available
            pass
        
    def test_all_classes_from_file_obsenv(self, env: Optional[Environment]=None):
        """test the files are correctly generated for the "forecast env" in the 
        environment even after a call to obs.reset() and obs.simulate()"""
        if not self._do_test_obs_env():
            self.skipTest("ObsEnv is not tested")
        env = self._aux_make_env(env)
        
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}")  
        
        # reset and check the same
        obs = env.reset()    
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}")  
        self.test_all_classes_from_file(env=obs._obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}")   
        
        # forecast and check the same
        try:
            obs.simulate(env.action_space())
            self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                            name_action_cls="CompleteAction_{}",
                                            name_observation_cls="CompleteObservation_{}")  
            self.test_all_classes_from_file(env=obs._obs_env,
                                            name_action_cls="CompleteAction_{}",
                                            name_observation_cls="CompleteObservation_{}")  
        except NoForecastAvailable:
            # cannot do this test if the "original" env is a _Forecast env:
            # for l2rpn_case14_sandbox only 1 step ahead forecast are available
            pass 
    
    def test_all_classes_from_file_env_cpy(self, env: Optional[Environment]=None):
        """test that when an environment is copied, then the copied env is consistent, 
        that it is consistent after a reset and that the forecast env is consistent"""
        if not self._do_test_copy():
            self.skipTest("Copy is not tested")
        env = self._aux_make_env(env)
        env_cpy = env.copy()
        self.test_all_classes_from_file(env=env_cpy)
        self.test_all_classes_from_file_env_after_reset(env=env_cpy)
        self.test_all_classes_from_file(env=env_cpy.observation_space.obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}"
                                        )     
        self.test_all_classes_from_file_obsenv(env=env_cpy)
    
    def test_all_classes_from_file_env_runner(self, env: Optional[Environment]=None):
        """this test, using the defined functions above that the runner is able to create a valid env"""
        if not self._do_test_runner():
            self.skipTest("Runner not tested")
        env = self._aux_make_env(env)
        runner = Runner(**env.get_params_for_runner())
        env_runner = runner.init_env()     
        self.test_all_classes_from_file(env=env_runner)
        self.test_all_classes_from_file_env_after_reset(env=env_runner)
        self.test_all_classes_from_file(env=env_runner.observation_space.obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}")     
        self.test_all_classes_from_file_obsenv(env=env_runner)
        
        # test the runner prevents the deletion of the tmp file where the classes are stored
        # path_cls = env._local_dir_cls
        # del env
        # assert os.path.exists(path_cls.name)
        env_runner = runner.init_env()     
        self.test_all_classes_from_file(env=env_runner)
        self.test_all_classes_from_file_env_after_reset(env=env_runner)
        self.test_all_classes_from_file(env=env_runner.observation_space.obs_env,
                                        name_action_cls="CompleteAction_{}",
                                        name_observation_cls="CompleteObservation_{}")     
        self.test_all_classes_from_file_obsenv(env=env_runner)
    
    def test_all_classes_from_file_runner_1ep(self, env: Optional[Environment]=None):
        """this test that the runner is able to "run" (one type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        if not self._do_test_runner():
            self.skipTest("Runner not tested")
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    self._aux_get_obs_cls().format(self.get_env_name()),
                                    self._aux_get_act_cls().format(self.get_env_name()),
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        runner.run(nb_episode=1,
                   max_iter=self.max_iter,
                   env_seeds=[0],
                   episode_id=[0])
    
    def test_all_classes_from_file_runner_2ep_seq(self, env: Optional[Environment]=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        if not self._do_test_runner():
            self.skipTest("Runner not tested")
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    self._aux_get_obs_cls().format(self.get_env_name()),
                                    self._aux_get_act_cls().format(self.get_env_name()),
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        res = runner.run(nb_episode=2,
                         max_iter=self.max_iter,
                         env_seeds=[0, 0],
                         episode_id=[0, 1])
        assert res[0][4] == self.max_iter
        assert res[1][4] == self.max_iter
    
    def test_all_classes_from_file_runner_2ep_par_fork(self, env: Optional[Environment]=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        if not self._do_test_runner():
            self.skipTest("Runner not tested")
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    self._aux_get_obs_cls().format(self.get_env_name()),
                                    self._aux_get_act_cls().format(self.get_env_name()),
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        res = runner.run(nb_episode=2,
                         nb_process=2,
                         max_iter=self.max_iter,
                         env_seeds=[0, 0],
                         episode_id=[0, 1])
        assert res[0][4] == self.max_iter
        assert res[1][4] == self.max_iter
    
    def test_all_classes_from_file_runner_2ep_par_spawn(self, env: Optional[Environment]=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        if not self._do_test_runner():
            self.skipTest("Runner not tested")
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    self._aux_get_obs_cls().format(self.get_env_name()),
                                    self._aux_get_act_cls().format(self.get_env_name()),
                                    )
        ctx = mp.get_context('spawn')
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent,
                        mp_context=ctx)
        res = runner.run(nb_episode=2,
                         nb_process=2,
                         max_iter=self.max_iter,
                         env_seeds=[0, 0],
                         episode_id=[0, 1])
        assert res[0][4] == self.max_iter
        assert res[1][4] == self.max_iter
        
        
class MaskedEnvAutoClassTester(AutoClassInFileTester):

    def _aux_make_env(self, env: Optional[Environment]=None):
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = MaskedEnvironment(super()._aux_make_env(),
                                        lines_of_interest=np.array([True, True, True, True, True, True,
                                                                    False, False, False, False, False, False,
                                                                    False, False, False, False, False, False,
                                                                    False, False]))
        return env
        
        
class TOEnvAutoClassTester(AutoClassInFileTester):

    def _aux_make_env(self, env: Optional[Environment]=None):
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = TimedOutEnvironment(super()._aux_make_env(),
                                          time_out_ms=1e-3)
        return env
        
        
class ForEnvAutoClassTester(AutoClassInFileTester):

    def _aux_make_env(self, env: Optional[Environment]=None):
        if env is None:
            # we create the reference environment and prevent grid2op to 
            # to delete it (because it stores the files to the class)
            self.ref_env = super()._aux_make_env()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                obs = self.ref_env.get_obs()
                res = obs.get_forecast_env()
            self.max_iter = res._max_iter  # otherwise it fails in the runner
        else:
            res = env
        return res
    
    def tearDown(self):
        if hasattr(self, "ref_env"):
            self.ref_env.close()
    

# class SEMPAUtoClassTester(AutoClassInFileTester):
#  """means i need to completely recode `test_all_classes_from_file` to take into account the return
#     values which is a list now... and i'm not ready for it yet TODO"""          
#     def _do_test_runner(self):
#         # false for multi process env
#         return False
    
#     def _do_test_copy(self):
#         # for for multi process env
#         return False
    
#     def _do_test_obs_env(self):
#         return False

#     def _aux_make_env(self, env: Optional[Environment]=None):
#         if env is None:
#             # we create the reference environment and prevent grid2op to 
#             # to delete it (because it stores the files to the class)
#             self.ref_env = super()._aux_make_env()
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore")
#                 res = SingleEnvMultiProcess(self.ref_env, nb_env=2)
#         else:
#             res = env
#         return res
    
class GymEnvAutoClassTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox",
                                    test=True,
                                    _add_to_name=type(self).__name__)
        self.line_id = 3
        th_lim = self.env.get_thermal_limit() * 2.  # avoid all problem in general
        th_lim[self.line_id] /= 10.  # make sure to get trouble in line 3
        self.env.set_thermal_limit(th_lim)
        
        GymEnvAutoClassTester._init_env(self.env)
        
    @staticmethod  
    def _init_env(env):
        env.set_id(0)
        env.seed(0)
        env.reset()

    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def _aux_run_envs(self, act, env_gym):
        for i in range(10):
            obs_in, reward, done, truncated, info = env_gym.step(act)
            if i < 2:  # 2 : 2 full steps already
                assert obs_in["timestep_overflow"][self.line_id] == i + 1, f"error for step {i}: {obs_in['timestep_overflow'][self.line_id]}"
            else:
                # cooldown applied for line 3: 
                # - it disconnect stuff in `self.env_in`
                # - it does not affect anything in `self.env_out`
                assert not obs_in["line_status"][self.line_id]
                
    def test_gym_with_step(self):
        """test the step function also disconnects (or not) the lines"""
        env_gym = GymEnv(self.env)
        act = {}
        self._aux_run_envs(act, env_gym)
        env_gym.reset()
        self._aux_run_envs(act, env_gym)
            
    def test_gym_normal(self):
        """test I can create the gym env"""
        env_gym = GymEnv(self.env)
        env_gym.reset()
    
    def test_gym_box(self):
        """test I can create the gym env with box ob space and act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = BoxGymActSpace(self.env.action_space)
            env_gym.observation_space = BoxGymObsSpace(self.env.observation_space)
        env_gym.reset()
    
    def test_gym_discrete(self):
        """test I can create the gym env with discrete act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = DiscreteActSpace(self.env.action_space)
        env_gym.reset()
        act = 0
        self._aux_run_envs(act, env_gym)
        
    def test_gym_multidiscrete(self):
        """test I can create the gym env with multi discrete act space"""
        env_gym = GymEnv(self.env)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_gym.action_space = MultiDiscreteActSpace(self.env.action_space)
        env_gym.reset()
        act = env_gym.action_space.sample()
        act[:] = 0
        self._aux_run_envs(act, env_gym)
        
    def test_asynch_fork(self):
        async_vect_env = AsyncVectorEnv((lambda: GymEnv(self.env), lambda: GymEnv(self.env)),
                                        context="fork")
        obs = async_vect_env.reset()
        
    def test_asynch_spawn(self):
        async_vect_env = AsyncVectorEnv((lambda: GymEnv(self.env), lambda: GymEnv(self.env)),
                                        context="spawn")
        obs = async_vect_env.reset()
        
        
class MultiMixEnvAutoClassTester(AutoClassInFileTester):
    def _aux_get_obs_cls(self):
        return "ObservationNeurips2020_{}"
    
    def _aux_get_act_cls(self):
        return "ActionNeurips2020_{}"
    
    def get_env_name(self):
        return "l2rpn_neurips_2020_track2"
    # TODO gym for that too
    
    # def _do_test_runner(self):
    #     return False
    
    def test_all_classes_from_file(self,
                                   env: Optional[Environment]=None,
                                   classes_name=None,
                                   name_complete_obs_cls="CompleteObservation_{}",
                                   name_observation_cls=None,
                                   name_action_cls=None):
        env = self._aux_make_env(env)
        super().test_all_classes_from_file(env,
                                           classes_name=classes_name,
                                           name_complete_obs_cls=name_complete_obs_cls,
                                           name_observation_cls=name_observation_cls,
                                           name_action_cls=name_action_cls
                                           )
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multi mix
            for mix in env:
                super().test_all_classes_from_file(mix,
                                                classes_name=classes_name,
                                                name_complete_obs_cls=name_complete_obs_cls,
                                                name_observation_cls=name_observation_cls,
                                                name_action_cls=name_action_cls
                                                )
    
    def test_all_classes_from_file_env_after_reset(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        super().test_all_classes_from_file_env_after_reset(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_env_after_reset(mix)
    
    def test_all_classes_from_file_obsenv(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        super().test_all_classes_from_file_obsenv(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_obsenv(mix)
    
    def test_all_classes_from_file_env_cpy(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        super().test_all_classes_from_file_env_cpy(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_env_cpy(mix)
                
    def test_all_classes_from_file_env_runner(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_env_runner(mix)
        else:
            # runner does not handle multimix
            super().test_all_classes_from_file_env_runner(env)  
                
    def test_all_classes_from_file_runner_1ep(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_runner_1ep(mix)
        else:
            # runner does not handle multimix
            super().test_all_classes_from_file_runner_1ep(env)  
                
    def test_all_classes_from_file_runner_2ep_seq(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_runner_2ep_seq(mix)
        else:
            # runner does not handle multimix
            super().test_all_classes_from_file_runner_2ep_seq(env)  
                
    def test_all_classes_from_file_runner_2ep_par_fork(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_runner_2ep_par_fork(mix)
        else:
            # runner does not handle multimix
            super().test_all_classes_from_file_runner_2ep_par_fork(env)  
                
    def test_all_classes_from_file_runner_2ep_par_spawn(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                super().test_all_classes_from_file_runner_2ep_par_spawn(mix)
        else:
            # runner does not handle multimix
            super().test_all_classes_from_file_runner_2ep_par_spawn(env)  
            
    def test_forecast_env_basic(self, env: Optional[Environment]=None):
        env = self._aux_make_env(env)
        if isinstance(env, MultiMixEnvironment):
            # test each mix of a multimix
            for mix in env:
                obs = mix.reset()
                for_env = obs.get_forecast_env()
                super().test_all_classes_from_file(for_env)

        
if __name__ == "__main__":
    unittest.main()
