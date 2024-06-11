# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
from grid2op.Exceptions import Grid2OpException
import unittest
import pdb


class InitTSOptions(unittest.TestCase):
    """test the "init ts" options in env.reset() """
    def setUp(self) -> None:
        self.env_name = "l2rpn_case14_sandbox"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name, test=True,
                                    _add_to_name=type(self).__name__)
            
    def test_function_ok(self):
        obs = self.env.reset() # normal reset
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 6
        assert obs.hour_of_day == 0
        assert obs.minute_of_hour == 0  
        
        obs = self.env.reset(options={"init ts": 1})  # skip the first step, start at 5 minutes
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 6
        assert obs.hour_of_day == 0
        assert obs.minute_of_hour == 5, f"{ obs.minute_of_hour} vs 5"
        
        obs = self.env.reset(options={"init ts": 2}) # start after 10 minutes, 2 steps
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 6
        assert obs.hour_of_day == 0
        assert obs.minute_of_hour == 10, f"{ obs.minute_of_hour} vs 10"
        
        obs = self.env.reset(options={"init ts": 6})  # start after 6steps (30 minutes)
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 6
        assert obs.hour_of_day == 0
        assert obs.minute_of_hour == 30, f"{ obs.minute_of_hour} vs 30"
        
        obs = self.env.reset(options={"init ts": 12})  # start at the 12th step
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 6
        assert obs.hour_of_day == 1, f"{ obs.minute_of_hour} vs 1"
        assert obs.minute_of_hour == 0, f"{ obs.minute_of_hour} vs 0"
        
        obs = self.env.reset(options={"init ts": 12 * 24})  # start after exactly 1 day
        assert obs.year == 2019
        assert obs.month == 1
        assert obs.day == 7, f"{ obs.day} vs 7"
        assert obs.hour_of_day == 0, f"{ obs.hour_of_day} vs 1"
        assert obs.minute_of_hour == 0, f"{ obs.minute_of_hour} vs 0"
    
    def check_soft_overflow(self):
        """check that the lines are not on soft overflow (obs.timestep_overflow == 0 just after reset)"""
        line_id = 3
        obs = self.env.reset(options={"time serie id": 0})
        th_lim = 1. * self.env.get_thermal_limit()
        th_lim[line_id] = 0.6 * obs.a_or[line_id]
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset(options={"time serie id": 0})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] > 1.
        assert obs.line_status[line_id]
        
        obs = self.env.reset(options={"time serie id": 0})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] > 1.
        assert obs.line_status[line_id]
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 1})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] > 1.
        assert obs.line_status[line_id]
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 2})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] > 1.
        assert obs.line_status[line_id]
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 6})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] > 1.
        assert obs.line_status[line_id]
    
    def check_hard_overflow(self):
        """check lines are disconnected if on hard overflow at the beginning"""
        line_id = 3
        obs = self.env.reset(options={"time serie id": 0})
        th_lim = 1. * self.env.get_thermal_limit()
        th_lim[line_id] = 0.4 * obs.a_or[line_id]
        self.env.set_thermal_limit(th_lim)
        obs = self.env.reset(options={"time serie id": 0})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] == 0.
        assert not obs.line_status[line_id]
        assert obs.time_before_cooldown_line[line_id] == 0
        
        obs = self.env.reset(options={"time serie id": 0})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] == 0.
        assert not obs.line_status[line_id]
        assert obs.time_before_cooldown_line[line_id] == 0
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 1})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] == 0.
        assert not obs.line_status[line_id]
        assert obs.time_before_cooldown_line[line_id] == 0
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 2})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] == 0.
        assert not obs.line_status[line_id]
        assert obs.time_before_cooldown_line[line_id] == 0
        
        obs = self.env.reset(options={"time serie id": 0, "init ts": 6})
        assert (obs.timestep_overflow == 0).all()
        assert obs.rho[line_id] == 0.
        assert not obs.line_status[line_id]
        assert obs.time_before_cooldown_line[line_id] == 0
        
    
    def test_raise_if_args_not_correct(self):
        with self.assertRaises(Grid2OpException):
            # string and not int
            obs = self.env.reset(options={"init ts": "treliug"})
        with self.assertRaises(Grid2OpException):
            # float which is not an int
            obs = self.env.reset(options={"init ts": 1.5})
            
        # should work with a float convertible to an int
        obs = self.env.reset(options={"time serie id": 0, "init ts": 6.})
    
    
if __name__ == "__main__":
    unittest.main()    
