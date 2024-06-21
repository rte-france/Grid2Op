# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
os.environ["grid2op_class_in_file"] = "true"

import sys
import warnings
import unittest
import importlib

import grid2op
from grid2op.MakeEnv.PathUtils import USE_CLASS_IN_FILE
assert USE_CLASS_IN_FILE

# TODO feature: in the make add a kwargs to deactivate this

# TODO test Multiprocess
# TODO test multi mix
# TODO test runner
# TODO test env copy

# TODO test gym
# TODO two envs same name => now diff classes
# TODO test the runner saved classes and reload
# TODO test add_to_name
# TODO test noshunt
# TODO test backend converters
# TODO mode to propagate the "pointer" (this_local_dir = tempfile.TemporaryDirectory(dir=sys_path))
#      in all copy of the environment instead of keeping it only for the first one
# TODO test all type of backend in the observation space, including the deactivate forecast, reactivate forecast, the different backend etc.

class AutoClassInFileTester(unittest.TestCase):
    def get_env_name(self):
        return "l2rpn_case14_sandbox"
    
    def test_class_env_from_file(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.get_env_name(), test=True)
            
        from Environment_l2rpn_case14_sandbox_file import Environment_l2rpn_case14_sandbox
        assert type(env) is Environment_l2rpn_case14_sandbox
    
    def test_all_classes_from_file(self,
                                   env=None,
                                   classes_name="l2rpn_case14_sandbox",
                                   name_action_cls="PlayableAction_l2rpn_case14_sandbox"):
        if env is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(self.get_env_name(), test=True)
        
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
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert env._observation_space.obs_env._observation_space.subtype is this_class
            elif name_cls == f"DontAct_{classes_name}":
                assert env._oppSpace.action_space.subtype is this_class
                assert env._opponent.action_space.subtype is this_class
            elif name_cls == f"_ObsEnv_{classes_name}":
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.observation_space.obs_env) is this_class
                    assert isinstance(env.observation_space.obs_env, this_class)
            elif name_cls == f"ObservationSpace_{classes_name}":
                if env.observation_space.obs_env is not None:
                    # not in _ObsEnv
                    assert type(env.observation_space.obs_env._observation_space) is this_class
            elif name_cls == name_action_cls:
                assert env._action_space.subtype is this_class
                # assert env.observation_space.obs_env._actionClass is this_class  # not it's a complete action apparently
            elif name_cls == f"VoltageOnlyAction_{classes_name}":
                if env._voltage_controler is not None:
                    # not in _ObsEnv
                    assert env._voltage_controler.action_space.subtype is this_class
                
    def test_all_classes_from_file_obsenv(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.get_env_name(), test=True)
        
        self.test_all_classes_from_file(env=env.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")      
    
    def test_all_classes_from_file_env_cpy(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.get_env_name(), test=True)
        env_cpy = env.copy()
        self.test_all_classes_from_file(env=env_cpy)
        self.test_all_classes_from_file(env=env_cpy.observation_space.obs_env,
                                        name_action_cls="CompleteAction_l2rpn_case14_sandbox")     
                

if __name__ == "__main__":
    unittest.main()
