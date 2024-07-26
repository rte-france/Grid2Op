# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import os
import unittest
import numpy as np

from grid2op.tests.helper_path_test import PATH_DATA_TEST
import grid2op

class TestFlexibility(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = os.path.join(PATH_DATA_TEST, "5bus_example_with_flexibility")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_name,
                test=True,
                _add_to_name=type(self).__name__
            )
        self.env.set_id(0)
        _ = self.env.reset()
        self.ref_obs, *_ = self.env.step(self.env.action_space())
        
        self.env.set_id(0)
        _ = self.env.reset()

        self.flex_max_ramp_up = self.env.action_space(
            {"flexibility": [(el, self.env.load_max_ramp_up[el]) for el in np.where(self.env.load_flexible)[0]]}
        )
        self.flex_max_ramp_down = self.env.action_space(
            {"flexibility": [(el, -self.env.load_max_ramp_down[el]) for el in np.where(self.env.load_flexible)[0]]}
        )
        self.flex_all_zero = self.env.action_space(
            {"flexibility": [(el, 0.0) for el in np.where(self.env.load_flexible)[0]]}
        )
        self.flex_small_up = self.env.action_space(
            {"flexibility": [(el, 0.01) for el in np.where(self.env.load_flexible)[0]]}
        )
        self.flex_small_down = self.env.action_space(
            {"flexibility": [(el, 0.01) for el in np.where(self.env.load_flexible)[0]]}
        )

    def test_zero_flex(self):
        flex_obs, *_ = self.env.step(self.flex_all_zero)
        flex_mask = self.env.load_flexible
        # Change in load relative to DoNothing scenario (i.e. normal Chronics)
        change_in_load = self.ref_obs.load_p[flex_mask] - flex_obs.load_p[flex_mask]
        assert np.isclose(change_in_load, np.zeros(flex_mask.sum()), atol=0.001).all()
    
    def test_flex_small_up(self):
        flex_obs, *_ = self.env.step(self.flex_small_up)
        flex_mask = self.env.load_flexible
        # Change in load relative to DoNothing scenario (i.e. normal Chronics)
        change_in_load = flex_obs.load_p[flex_mask] - self.ref_obs.load_p[flex_mask]
        assert np.isclose(change_in_load, self.flex_small_up.flexibility[flex_mask], atol=0.001).all()

    def test_flex_small_down(self):
        flex_obs, *_  = self.env.step(self.flex_small_down)
        flex_mask = self.env.load_flexible

        # Change in load relative to DoNothing scenario (i.e. normal Chronics)
        change_in_load = flex_obs.load_p[flex_mask] - self.ref_obs.load_p[flex_mask]
        assert np.isclose(change_in_load, self.flex_small_down.flexibility[flex_mask], atol=0.001).all()

    def test_flex_max_ramp_up(self):
        flex_obs, *_ = self.env.step(self.flex_max_ramp_up)
        flex_mask = self.env.load_flexible
        # Load meets max ramp up, or the max size of the load
        ref_load = self.ref_obs.load_p[flex_mask]
        expected_load = ref_load + self.flex_max_ramp_up.flexibility[flex_mask]
        maximum_feasible_load = np.minimum(self.env.load_size[flex_mask], expected_load)
        assert np.isclose(flex_obs.load_p[flex_mask], maximum_feasible_load, atol=0.001).all()

    def test_flex_max_ramp_down(self):
        flex_obs, *_ = self.env.step(self.flex_max_ramp_down)
        flex_mask = self.env.load_flexible
        # Load meets max ramp down, or the minimum load (of 0)
        ref_load = self.ref_obs.load_p[flex_mask]
        expected_load = ref_load + self.flex_max_ramp_down.flexibility[flex_mask]
        minimum_feasible_load = np.maximum(np.zeros(flex_mask.sum()), expected_load)
        assert np.isclose(flex_obs.load_p[flex_mask], minimum_feasible_load, atol=0.001).all()

if __name__ == "__main__":
    unittest.main()
