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
from grid2op.Environment import MaskedEnvironment, TimedOutEnvironment
from grid2op.Backend import PandaPowerBackend
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
        
    def test_regular_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, 2)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
    def test_multimix_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_2")
        self._aux_fun_test(env, 2)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track2", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_3")
        self._aux_fun_test(env, 3)
        
    def test_masked_env(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = MaskedEnvironment(grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_mask_2"),
                                    lines_of_interest=np.ones(shape=20, dtype=bool))
        self._aux_fun_test(env, 2)
        
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
        self._aux_fun_test(env, 2)
        
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
        self._aux_fun_test(env, 2)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoCalled(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_nocall_3")
        self._aux_fun_test(env, 2)
    
    def test_cannot_handle_more_than_2_busbar_not_called(self):
        """when using a backend that called `cannot_handle_more_than_2_busbar_not_called` then it's equivalent 
        to not support this new feature."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, _add_to_name=type(self).__name__+"_dontcalled_2")
        self._aux_fun_test(env, 2)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendNoSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_dontcalled_3")
        self._aux_fun_test(env, 2)
        
    def test_env_copy(self):
        """test env copy does work correctly"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, _add_to_name=type(self).__name__+"_copy_2")
        self._aux_fun_test(env, 2)
        env_cpy = env.copy()
        self._aux_fun_test(env_cpy, 2)
        
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
        self._aux_fun_test(env_2, 2)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_3 = grid2op.make("l2rpn_case14_sandbox", backend=_AuxFakeBackendSupport(), test=True, n_busbar=3, _add_to_name=type(self).__name__+"_same_name")
        self._aux_fun_test(env_3, 3)  # check env_3 has indeed 3 buses
        self._aux_fun_test(env_2, 2)  # check env_2 is not modified
        
        