# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
import unittest

import grid2op
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
import numpy as np


class Issue321Tester(unittest.TestCase):
    def test_curtail_limit_effective(self):
        """test that obs.curtailment_limit is above 
        obs.curtailment_limit_effective in this case
        
        previously it was O. for some generators see the issue
        """
        param = Parameters()
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        env_name  = "educ_case14_storage"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(env_name, test=True,
                               action_class=PlayableAction, param=param,
                               _add_to_name=type(self).__name__)
            
        env.seed(0)
        env.set_id(0)
        env.reset()

        # Apply a curtailment action
        action = env.action_space({'curtail': [(2, 0.1), (3, 0.15), (4, 0.2)]})
        obs, _, _, _ = env.step(action)
        # env does not act, they should be equal
        assert np.all(obs.curtailment_limit == obs.curtailment_limit_effective)

    def test_when_params_active_cutdown(self):
        """test the curtailment effective is above the curtailment 
        when we want to curtail too much at once"""
        seed_ = 0
        scen_nm = "2050-02-14_0"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_wcci_2022_dev",
                                    test=True,
                                    _add_to_name=type(self).__name__)
        param = env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        env.seed(seed_)
        env.set_id(scen_nm)
        obs = env.reset()
        
        all_zero = env.action_space(
            {"curtail": [(el, 0.0) for el in np.where(env.gen_renewable)[0]]}
        )
        obs, reward, done, info = env.step(all_zero)
        assert done
        
        env.change_parameters(param)
        env.seed(seed_)
        env.set_id(scen_nm)
        obs = env.reset()
        obs, reward, done, info = env.step(all_zero)
        assert not done
        # stuff is activated, the real limit (effective) is above the
        # limit in the action (env "cut down" curtailment)
        assert np.all(obs.curtailment_limit <= obs.curtailment_limit_effective)
        assert np.all(obs.curtailment_limit[env.gen_renewable] == 0.)

    def test_when_params_active_limitup(self):
        """test the curtailment effective is below the curtailment limit
        when too much curtailment has been removed "at once"
        """
        seed_ = 0
        scen_nm = "2050-02-14_0"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_wcci_2022_dev",
                                    test=True,
                                    _add_to_name=type(self).__name__)
        param = env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        env.seed(seed_)
        env.set_id(scen_nm)
        obs = env.reset()
        
        # produce a list of actions that can curtail down to O.
        # then remove all curtailment at once (and check it
        # breaks the env)
        first_ = env.action_space(
            {"curtail": [(el, 0.15) for el in np.where(env.gen_renewable)[0]]}
        )
        second_ = env.action_space(
            {"curtail": [(el, 0.09) for el in np.where(env.gen_renewable)[0]]}
        )
        third_ = env.action_space(
            {"curtail": [(el, 0.04) for el in np.where(env.gen_renewable)[0]]}
        )
        all_zero = env.action_space(
            {"curtail": [(el, 0.0) for el in np.where(env.gen_renewable)[0]]}
        )
        all_one = env.action_space(
            {"curtail": [(el, 1.0) for el in np.where(env.gen_renewable)[0]]}
        )
        obs, reward, done, info = env.step(first_)
        assert not done
        obs, reward, done, info = env.step(second_)
        assert not done
        obs, reward, done, info = env.step(third_)
        assert not done
        obs, reward, done, info = env.step(all_zero)
        assert not done
        obs, reward, done, info = env.step(all_one)
        assert done
        
        # use this same list of actions, and make sure it does not break the
        # env with the appropriate params (expected !)
        # => this means the params is effective. And so the env prevent a 
        # removal of the curtailment too strong.
        env.change_parameters(param)
        env.seed(seed_)
        env.set_id(scen_nm)
        obs = env.reset()
        obs, reward, done, info = env.step(first_)
        assert not done
        obs, reward, done, info = env.step(second_)
        assert not done
        obs, reward, done, info = env.step(third_)
        assert not done
        obs, reward, done, info = env.step(all_zero)
        assert not done
        obs, reward, done, info = env.step(all_one)
        assert not done
        # stuff is activated, the real limit (effective) is below the
        # limit in the action: instead of having a curtailment of 1.,
        # curtailment is a below, just enough not to break
        assert np.all(obs.curtailment_limit[env.gen_renewable] == 1.)
        gen_prod = obs.gen_p > 0.
        assert np.all(obs.curtailment_limit[env.gen_renewable & gen_prod] > 
                      obs.curtailment_limit_effective[env.gen_renewable & gen_prod])
        
if __name__ == "__main__":
    unittest.main()
       