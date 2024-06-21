# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import multiprocessing as mp

os.environ["grid2op_class_in_file"] = "true"

import sys
import warnings
import unittest
import importlib

import grid2op
from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
from grid2op.Runner import Runner
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Action.actionSpace import ActionSpace
assert USE_CLASS_IN_FILE

# TODO feature: in the make add a kwargs to deactivate this

# TODO test Multiprocess
# TODO test multi mix
# TODO test runner

# TODO test gym
# TODO two envs same name => now diff classes
# TODO test the runner saved classes and reload
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


class AutoClassInFileTester(unittest.TestCase):
    def get_env_name(self):
        return "l2rpn_case14_sandbox"
    
    def setUp(self) -> None:
        self.max_iter = 10
        return super().setUp()
    
    def _aux_make_env(self, env=None):
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(self.get_env_name(), test=True)
        return env
        
    def test_all_classes_from_file(self,
                                   env=None,
                                   classes_name="l2rpn_case14_sandbox",
                                   name_action_cls="PlayableAction_l2rpn_case14_sandbox"):
        env = self._aux_make_env(env)
        names_cls = [f"ActionSpace_{classes_name}",
                     f"_BackendAction_{classes_name}",
                     f"CompleteAction_{classes_name}",
                     f"CompleteObservation_{classes_name}",
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
                      None, # DONT ACT not int ENV directlu
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
                    assert the_attr is this_class, f"error for {name_cls} (env.{name_attr})"
                else:
                    assert type(the_attr) is this_class, f"error for {name_cls} (env.{name_attr})"
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
            elif name_cls == f"CompleteObservation_{classes_name}":
                assert env._observation_space.subtype is this_class
                if env.current_obs is not None:
                    # not in _ObsEnv
                    assert isinstance(env.current_obs, this_class)
                if env._last_obs is not None:
                    # not in _ObsEnv
                    assert isinstance(env._last_obs, this_class)
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
        
    def test_all_classes_from_file_env_after_reset(self, env=None):
        """test classes are still consistent even after a call to env.reset() and obs.simulate()"""
        env = self._aux_make_env(env)
        obs = env.reset()
        self.test_all_classes_from_file(env=env)
        obs.simulate(env.action_space())
        self.test_all_classes_from_file(env=env)
        
    def test_all_classes_from_file_obsenv(self, env=None):
        """test the files are correctly generated for the "forecast env" in the 
        environment even after a call to obs.reset() and obs.simulate()"""
        env = self._aux_make_env(env)
        
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")  
        
        # reset and check the same
        obs = env.reset()    
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")  
        self.test_all_classes_from_file(env=obs._obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")   
        
        # forecast and check the same
        obs.simulate(env.action_space())
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")  
        self.test_all_classes_from_file(env=obs._obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")   
    
    def test_all_classes_from_file_env_cpy(self, env=None):
        """test that when an environment is copied, then the copied env is consistent, 
        that it is consistent after a reset and that the forecast env is consistent"""
        env = self._aux_make_env(env)
        env_cpy = env.copy()
        self.test_all_classes_from_file(env=env_cpy)
        self.test_all_classes_from_file_env_after_reset(env=env_cpy)
        self.test_all_classes_from_file(env=env_cpy.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")     
        self.test_all_classes_from_file_obsenv(env=env_cpy)
    
    def test_all_classes_from_file_env_runner(self, env=None):
        """this test, using the defined functions above that the runner is able to create a valid env"""
        env = self._aux_make_env(env)
        runner = Runner(**env.get_params_for_runner())
        env_runner = runner.init_env()     
        self.test_all_classes_from_file(env=env_runner)
        self.test_all_classes_from_file_env_after_reset(env=env_runner)
        self.test_all_classes_from_file(env=env_runner.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")     
        self.test_all_classes_from_file_obsenv(env=env_runner)
        
        # test the runner prevents the deletion of the tmp file where the classes are stored
        # path_cls = env._local_dir_cls
        # del env
        # assert os.path.exists(path_cls.name)
        env_runner = runner.init_env()     
        self.test_all_classes_from_file(env=env_runner)
        self.test_all_classes_from_file_env_after_reset(env=env_runner)
        self.test_all_classes_from_file(env=env_runner.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")     
        self.test_all_classes_from_file_obsenv(env=env_runner)
    
    def test_all_classes_from_file_runner_1ep(self, env=None):
        """this test that the runner is able to "run" (one type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    f"CompleteObservation_{self.get_env_name()}",
                                    f"PlayableAction_{self.get_env_name()}",
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        runner.run(nb_episode=1,
                   max_iter=self.max_iter,
                   env_seeds=[0],
                   episode_id=[0])
    
    def test_all_classes_from_file_runner_2ep_seq(self, env=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    f"CompleteObservation_{self.get_env_name()}",
                                    f"PlayableAction_{self.get_env_name()}",
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        runner.run(nb_episode=2,
                   max_iter=self.max_iter,
                   env_seeds=[0, 0],
                   episode_id=[0, 1])
    
    def test_all_classes_from_file_runner_2ep_par_fork(self, env=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    f"CompleteObservation_{self.get_env_name()}",
                                    f"PlayableAction_{self.get_env_name()}",
                                    )
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent)
        runner.run(nb_episode=2,
                   nb_process=2,
                   max_iter=self.max_iter,
                   env_seeds=[0, 0],
                   episode_id=[0, 1])
    
    def test_all_classes_from_file_runner_2ep_par_spawn(self, env=None):
        """this test that the runner is able to "run" (one other type of run), but the tests on the classes 
        are much lighter than in test_all_classes_from_file_env_runner"""
        env = self._aux_make_env(env)
        this_agent = _ThisAgentTest(env.action_space,
                                    env._read_from_local_dir,
                                    f"CompleteObservation_{self.get_env_name()}",
                                    f"PlayableAction_{self.get_env_name()}",
                                    )
        ctx = mp.get_context('spawn')
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=None,
                        agentInstance=this_agent,
                        mp_context=ctx)
        runner.run(nb_episode=2,
                   nb_process=2,
                   max_iter=self.max_iter,
                   env_seeds=[0, 0],
                   episode_id=[0, 1])
        
        
if __name__ == "__main__":
    unittest.main()
