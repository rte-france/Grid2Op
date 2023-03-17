# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
import unittest
import pdb

from grid2op.Backend import PandaPowerBackend
from grid2op.Runner import Runner


class InstanceCount:
    """just to make it work... does not really work"""
    nb_instance = 0
    def __init__(self) -> None:
        type(self).nb_instance += 1
    def __del__(self):
        type(self).nb_instance -= 1
        
        
class PPExtraArgs(PandaPowerBackend):
    def __init__(self,
                 stuff="",
                 detailed_infos_for_cascading_failures=False,
                 lightsim2grid=False,
                 dist_slack=False,
                 max_iter=10,
                 can_be_copied=True):
        super().__init__(detailed_infos_for_cascading_failures,
                         lightsim2grid,
                         dist_slack,
                         max_iter,
                         can_be_copied=can_be_copied)
        self._my_kwargs["stuff"] = stuff


class BackendProperlyInit(unittest.TestCase):
    """test grid2op works when the backend cannot be copied."""
    def setUp(self) -> None:
        self.env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name, test=True, backend=PPExtraArgs())
            
    def tearDown(self) -> None:
        self.env.close()
        
    def test_default_args(self):
        """basic tests, with default arguments"""
        kwargs = self.env.get_params_for_runner()
        assert "backend_kwargs" in kwargs
        runner = Runner(**kwargs)
        assert runner._backend_kwargs == kwargs["backend_kwargs"]
        env = runner.init_env()
        assert "stuff" in env.backend._my_kwargs
        assert env.backend._my_kwargs["stuff"] == ""
        env.close()
        
    def test_non_default_args(self):
        """test with non default args: they are used properly in the runner"""
        self.env.close()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name,
                                    test=True,
                                    backend=PPExtraArgs(stuff="toto"))
        runner = Runner(**self.env.get_params_for_runner())
        env = runner.init_env()
        assert env.backend._my_kwargs["stuff"] == "toto"
        env.close()
    
    def test_make_no_copy(self):
        """test that it does not make any copy of the default arguments"""
        self.env.close()
        counter = InstanceCount()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name,
                                    test=True,
                                    backend=PPExtraArgs(stuff=counter))
        runner = Runner(**self.env.get_params_for_runner())
        env = runner.init_env()
        assert isinstance(env.backend._my_kwargs["stuff"], InstanceCount)
        assert env.backend._my_kwargs["stuff"] is counter
        assert type(env.backend._my_kwargs["stuff"]).nb_instance == 1
        env.close()
        

if __name__ == "__main__":
    unittest.main()
