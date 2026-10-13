# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import hashlib
import unittest
import warnings

import numpy as np

import grid2op
from grid2op.Chronics import FromNPY
from grid2op.Exceptions import ChronicsError

N_TS = 6
N_HORIZONS = 3
N_LOAD = 4
N_GEN = 2


def _arr(n_col, n_h=None, value=1.0):
    shape = (N_TS, n_col) if n_h is None else (N_TS, n_h, n_col)
    return value * np.ones(shape)


def _make(**kwargs):
    base = dict(
        load_p=_arr(N_LOAD),
        load_q=_arr(N_LOAD),
        prod_p=_arr(N_GEN),
        prod_v=_arr(N_GEN),
    )
    base.update(kwargs)
    return FromNPY(**base)


def _multi(**kwargs):
    base = dict(
        load_p_forecast=_arr(N_LOAD, N_HORIZONS),
        load_q_forecast=_arr(N_LOAD, N_HORIZONS),
        prod_p_forecast=_arr(N_GEN, N_HORIZONS),
    )
    base.update(kwargs)
    return _make(**base)


class TestFromNPYForecastArgumentsErrors(unittest.TestCase):
    """The forecast arguments given to the constructor are checked."""

    def test_load_p_forecast_needs_load_q_forecast(self):
        with self.assertRaises(ChronicsError):
            _make(load_p_forecast=_arr(N_LOAD), prod_p_forecast=_arr(N_GEN))

    def test_load_p_forecast_needs_prod_p_forecast(self):
        with self.assertRaises(ChronicsError):
            _make(load_p_forecast=_arr(N_LOAD), load_q_forecast=_arr(N_LOAD))

    def test_load_q_forecast_needs_load_p_forecast(self):
        with self.assertRaises(ChronicsError):
            _make(load_q_forecast=_arr(N_LOAD))

    def test_prod_p_forecast_needs_load_p_forecast(self):
        with self.assertRaises(ChronicsError):
            _make(prod_p_forecast=_arr(N_GEN))

    def test_mixed_dimensions_2d_first(self):
        with self.assertRaises(ChronicsError):
            _make(
                load_p_forecast=_arr(N_LOAD),
                load_q_forecast=_arr(N_LOAD, N_HORIZONS),
                prod_p_forecast=_arr(N_GEN),
            )
        with self.assertRaises(ChronicsError):
            _make(
                load_p_forecast=_arr(N_LOAD),
                load_q_forecast=_arr(N_LOAD),
                prod_p_forecast=_arr(N_GEN),
                prod_v_forecast=_arr(N_GEN, N_HORIZONS),
            )

    def test_unsupported_number_of_dimensions(self):
        with self.assertRaises(ChronicsError):
            _make(
                load_p_forecast=np.ones((N_TS, N_HORIZONS, 2, N_LOAD)),
                load_q_forecast=_arr(N_LOAD),
                prod_p_forecast=_arr(N_GEN),
            )

    def test_mixed_dimensions_3d_first(self):
        with self.assertRaises(ChronicsError):
            _multi(load_q_forecast=_arr(N_LOAD))

    def test_wrong_number_of_timesteps(self):
        for key, n_col in (
            ("load_p_forecast", N_LOAD),
            ("load_q_forecast", N_LOAD),
            ("prod_p_forecast", N_GEN),
            ("prod_v_forecast", N_GEN),
        ):
            arr = np.ones((N_TS + 1, N_HORIZONS, n_col))
            with self.subTest(key=key), self.assertRaises(ChronicsError):
                _multi(**{key: arr})

    def test_wrong_number_of_horizons(self):
        for key, n_col in (
            ("load_q_forecast", N_LOAD),
            ("prod_p_forecast", N_GEN),
            ("prod_v_forecast", N_GEN),
        ):
            arr = np.ones((N_TS, N_HORIZONS + 1, n_col))
            with self.subTest(key=key), self.assertRaises(ChronicsError):
                _multi(**{key: arr})

    def test_wrong_number_of_columns(self):
        for key, n_col in (
            ("load_p_forecast", N_LOAD),
            ("load_q_forecast", N_LOAD),
            ("prod_p_forecast", N_GEN),
            ("prod_v_forecast", N_GEN),
        ):
            arr = np.ones((N_TS, N_HORIZONS, n_col + 1))
            with self.subTest(key=key), self.assertRaises(ChronicsError):
                _multi(**{key: arr})


class TestFromNPYMultiHorizonForecasts(unittest.TestCase):
    """Behaviour of 3-dimensional ``(T, H, N)`` forecasts."""

    def test_forecasts_per_horizon(self):
        load_p_forecast = np.arange(N_TS * N_HORIZONS * N_LOAD, dtype=float).reshape(
            N_TS, N_HORIZONS, N_LOAD
        )
        chronics = _multi(
            load_p_forecast=load_p_forecast,
            prod_v_forecast=_arr(N_GEN, N_HORIZONS, 2.0),
        )
        self.assertEqual(chronics.n_forecast_horizons, N_HORIZONS)

        chronics.current_index = 1
        chronics.current_datetime = chronics.start_datetime
        res = chronics.forecasts()
        self.assertEqual(len(res), N_HORIZONS)
        for h, (dt, dict_) in enumerate(res):
            self.assertEqual(dt, chronics.start_datetime + (h + 1) * chronics.time_interval)
            inj = dict_["injection"]
            self.assertTrue(np.array_equal(inj["load_p"], load_p_forecast[1, h]))
            self.assertTrue(np.all(inj["prod_v"] == 2.0))

    def test_forecasts_without_prod_v(self):
        chronics = _multi()
        chronics.current_index = 0
        chronics.current_datetime = chronics.start_datetime
        for _, dict_ in chronics.forecasts():
            self.assertNotIn("prod_v", dict_["injection"])

    def test_forecasts_outside_of_the_data_are_empty(self):
        chronics = _multi()
        chronics.current_datetime = chronics.start_datetime
        chronics.current_index = -1
        self.assertEqual(chronics.forecasts(), [])
        chronics.current_index = N_TS
        self.assertEqual(chronics.forecasts(), [])

    def test_long_hash_depends_on_forecasts(self):
        def digest(chronics):
            return chronics._get_long_hash(hashlib.blake2b())

        ref = digest(_multi())
        self.assertEqual(ref, digest(_multi()))
        self.assertNotEqual(ref, digest(_multi(load_p_forecast=_arr(N_LOAD, N_HORIZONS, 3.0))))
        self.assertNotEqual(ref, digest(_multi(prod_v_forecast=_arr(N_GEN, N_HORIZONS))))


class TestFromNPYChangeForecasts(unittest.TestCase):
    """``change_forecasts`` with multi-horizon arrays (effective after ``reset``)."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_ref = grid2op.make(
                "l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__ + "ref"
            )
        data = env_ref.chronics_handler.real_data.data
        self.load_p = 1.0 * data.load_p
        self.load_q = 1.0 * data.load_q
        self.prod_p = 1.0 * data.prod_p
        self.prod_v = 1.0 * data.prod_v
        env_ref.close()
        self.n_ts = self.load_p.shape[0]
        self.n_load = self.load_p.shape[1]
        self.n_gen = self.prod_p.shape[1]

    def _forecast_arrays(self, n_h, value):
        return (
            value * np.ones((self.n_ts, n_h, self.n_load)),
            value * np.ones((self.n_ts, n_h, self.n_load)),
            value * np.ones((self.n_ts, n_h, self.n_gen)),
            value * np.ones((self.n_ts, n_h, self.n_gen)),
        )

    def _make_env(self, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return grid2op.make(
                "l2rpn_case14_sandbox",
                test=True,
                chronics_class=FromNPY,
                data_feeding_kwargs=dict(
                    load_p=self.load_p,
                    load_q=self.load_q,
                    prod_p=self.prod_p,
                    prod_v=self.prod_v,
                    **kwargs,
                ),
                _add_to_name=type(self).__name__,
            )

    def test_switch_from_no_forecast_to_multi_horizon(self):
        env = self._make_env()
        try:
            env.reset()
            chronics = env.chronics_handler.real_data
            self.assertEqual(chronics.forecasts(), [])
            chronics.change_forecasts(*self._forecast_arrays(2, 5.0))
            env.reset()
            chronics = env.chronics_handler.real_data
            self.assertEqual(chronics.n_forecast_horizons, 2)
            self.assertEqual(len(chronics.forecasts()), 2)
        finally:
            env.close()

    def test_update_multi_horizon_forecasts(self):
        env = self._make_env(
            **dict(
                zip(
                    ("load_p_forecast", "load_q_forecast", "prod_p_forecast", "prod_v_forecast"),
                    self._forecast_arrays(2, 1.0),
                )
            )
        )
        try:
            env.reset()
            chronics = env.chronics_handler.real_data
            self.assertEqual(chronics.n_forecast_horizons, 2)
            # 2d arrays are accepted and become a single horizon
            new_load_p, new_load_q, new_prod_p, new_prod_v = (
                a[:, 0, :] for a in self._forecast_arrays(1, 7.0)
            )
            chronics.change_forecasts(new_load_p, new_load_q, new_prod_p, new_prod_v)
            env.reset()
            chronics = env.chronics_handler.real_data
            self.assertEqual(chronics.n_forecast_horizons, 1)
            self.assertTrue(np.all(chronics._load_p_forecast == 7.0))
            self.assertTrue(np.all(chronics._prod_v_forecast == 7.0))
        finally:
            env.close()

    def test_incomplete_multi_horizon_change_is_refused(self):
        env = self._make_env()
        try:
            env.reset()
            load_p, load_q, prod_p, _ = self._forecast_arrays(2, 1.0)
            with self.assertRaises(ChronicsError):
                env.chronics_handler.real_data.change_forecasts(load_p, load_q, None)
        finally:
            env.close()

    def test_change_forecasts_without_forecasts_is_refused(self):
        env = self._make_env()
        try:
            env.reset()
            with self.assertRaises(ChronicsError):
                env.chronics_handler.real_data.change_forecasts(
                    new_load_p=self.load_p
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
