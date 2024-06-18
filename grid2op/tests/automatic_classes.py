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
    
    def test_all_classes_from_file(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(self.get_env_name(), test=True)
        
        names_cls = ["ActionSpace_l2rpn_case14_sandbox",
                     "_BackendAction_l2rpn_case14_sandbox",
                     "CompleteAction_l2rpn_case14_sandbox",
                     "CompleteObservation_l2rpn_case14_sandbox",
                    #  "DontAct_l2rpn_case14_sandbox",
                    #  "_ObsEnv_l2rpn_case14_sandbox",
                     "ObservationSpace_l2rpn_case14_sandbox",
                     "PandaPowerBackend_l2rpn_case14_sandbox",
                     "PlayableAction_l2rpn_case14_sandbox",
                    #  "VoltageOnlyAction_l2rpn_case14_sandbox"
                     ]
        names_attr = ["action_space",
                      "_backend_action_class",
                      "_complete_action_cls",
                      "_observationClass",
                      # DONT ACT not int ENV directlu
                    #   ObsEnv NOT IN ENV,
                      "observation_space",
                      "backend",
                      "_actionClass",
                      # VoltageOnlyAction_l2rpn_case14_sandbox not in env
                      ]
        for name_cls, name_attr in zip(names_cls, names_attr):
            this_module = importlib.import_module(f"{name_cls}_file", env._read_from_local_dir)
            if hasattr(this_module, name_cls):
                this_class = getattr(this_module, name_cls)
            else:
                raise RuntimeError(f"class {name_cls} not found")
            the_attr = getattr(env, name_attr)
            if isinstance(the_attr, type):
                assert the_attr is this_class, f"error for {name_cls} (env.{name_attr})"
            else:
                assert type(the_attr) is this_class, f"error for {name_cls} (env.{name_attr})"
            assert this_class._CLS_DICT is not None
            assert this_class._CLS_DICT_EXTENDED is not None
            # additional check for some attributes
            if name_cls == "_BackendAction_l2rpn_case14_sandbox":
                assert env.backend.my_bk_act_class is this_class
                assert isinstance(env._backend_action, this_class)
                
                assert env.observation_space.obs_env._backend_action_class is this_class
                assert env.observation_space.obs_env.backend.my_bk_act_class is this_class
                assert isinstance(env.observation_space.obs_env._backend_action, this_class)
            # TODO action space, observation space, opponent
        

if __name__ == "__main__":
    unittest.main()
