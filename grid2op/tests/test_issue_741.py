# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
from datetime import timedelta

import numpy as np

from grid2op.Chronics import FromNPY
from grid2op.Exceptions import ChronicsError


class TestIssue741MultiHorizonForecasts(unittest.TestCase):
    def setUp(self):
        self.timesteps = 4
        self.load_p = np.arange(8, dtype=float).reshape(4, 2) + 10.0
        self.load_q = self.load_p + 100.0
        self.prod_p = np.arange(4, dtype=float).reshape(4, 1) + 200.0
        self.prod_v = np.arange(4, dtype=float).reshape(4, 1) + 300.0

    def _make_chronics(self, load_p_fc, load_q_fc, prod_p_fc, prod_v_fc=None):
        chronics = FromNPY(
            load_p=self.load_p,
            load_q=self.load_q,
            prod_p=self.prod_p,
            prod_v=self.prod_v,
            load_p_forecast=load_p_fc,
            load_q_forecast=load_q_fc,
            prod_p_forecast=prod_p_fc,
            prod_v_forecast=prod_v_fc,
            time_interval=timedelta(minutes=5),
        )
        chronics.initialize(
            order_backend_loads=["load_0", "load_1"],
            order_backend_prods=["gen_0"],
            order_backend_lines=["line_0"],
            order_backend_subs=["sub_0"],
        )
        return chronics

    def test_two_dimensional_forecasts_keep_one_step_behavior(self):
        load_p_fc = self.load_p + 1.0
        load_q_fc = self.load_q + 1.0
        prod_p_fc = self.prod_p + 1.0
        prod_v_fc = self.prod_v + 1.0

        chronics = self._make_chronics(
            load_p_fc, load_q_fc, prod_p_fc, prod_v_fc
        )
        chronics.load_next()
        forecasts = chronics.forecasts()

        self.assertEqual(len(forecasts), 1)
        np.testing.assert_allclose(
            forecasts[0][1]["injection"]["load_p"], load_p_fc[0]
        )
        np.testing.assert_allclose(
            forecasts[0][1]["injection"]["load_q"], load_q_fc[0]
        )
        np.testing.assert_allclose(
            forecasts[0][1]["injection"]["prod_p"], prod_p_fc[0]
        )

    def test_three_dimensional_forecasts_return_each_horizon(self):
        horizons = 3
        load_p_fc = np.stack(
            [self.load_p + float(h + 1) for h in range(horizons)], axis=1
        )
        load_q_fc = np.stack(
            [self.load_q + float(h + 1) for h in range(horizons)], axis=1
        )
        prod_p_fc = np.stack(
            [self.prod_p + float(h + 1) for h in range(horizons)], axis=1
        )
        prod_v_fc = np.stack(
            [self.prod_v + float(h + 1) for h in range(horizons)], axis=1
        )

        chronics = self._make_chronics(
            load_p_fc, load_q_fc, prod_p_fc, prod_v_fc
        )
        chronics.load_next()
        forecasts = chronics.forecasts()

        self.assertEqual(len(forecasts), horizons)
        for h, (_, data) in enumerate(forecasts):
            np.testing.assert_allclose(
                data["injection"]["load_p"], load_p_fc[0, h]
            )
            np.testing.assert_allclose(
                data["injection"]["load_q"], load_q_fc[0, h]
            )
            np.testing.assert_allclose(
                data["injection"]["prod_p"], prod_p_fc[0, h]
            )
            np.testing.assert_allclose(
                data["injection"]["prod_v"], prod_v_fc[0, h]
            )

    def test_mismatched_horizon_count_is_rejected(self):
        load_p_fc = np.zeros((self.timesteps, 2, 2))
        load_q_fc = np.zeros((self.timesteps, 3, 2))
        prod_p_fc = np.zeros((self.timesteps, 2, 1))

        with self.assertRaises(ChronicsError):
            FromNPY(
                load_p=self.load_p,
                load_q=self.load_q,
                prod_p=self.prod_p,
                prod_v=self.prod_v,
                load_p_forecast=load_p_fc,
                load_q_forecast=load_q_fc,
                prod_p_forecast=prod_p_fc,
            )

    def test_change_forecasts_applies_after_reset(self):
        old_load_p_fc = self.load_p + 1.0
        old_load_q_fc = self.load_q + 1.0
        old_prod_p_fc = self.prod_p + 1.0

        chronics = self._make_chronics(
            old_load_p_fc, old_load_q_fc, old_prod_p_fc
        )
        chronics.load_next()

        horizons = 2
        new_load_p_fc = np.stack(
            [self.load_p + 10.0, self.load_p + 20.0], axis=1
        )
        new_load_q_fc = np.stack(
            [self.load_q + 10.0, self.load_q + 20.0], axis=1
        )
        new_prod_p_fc = np.stack(
            [self.prod_p + 10.0, self.prod_p + 20.0], axis=1
        )

        chronics.change_forecasts(
            new_load_p=new_load_p_fc,
            new_load_q=new_load_q_fc,
            new_prod_p=new_prod_p_fc,
        )

        self.assertEqual(len(chronics.forecasts()), 1)

        chronics.next_chronics()
        chronics.initialize(
            order_backend_loads=["load_0", "load_1"],
            order_backend_prods=["gen_0"],
            order_backend_lines=["line_0"],
            order_backend_subs=["sub_0"],
        )
        chronics.load_next()
        forecasts = chronics.forecasts()

        self.assertEqual(len(forecasts), horizons)
        np.testing.assert_allclose(
            forecasts[1][1]["injection"]["load_p"], new_load_p_fc[0, 1]
        )


if __name__ == "__main__":
    unittest.main()
