# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import ABC, abstractmethod
import unittest
import warnings

from grid2op.Exceptions import *
from grid2op.PlotGrid import *


class BaseTestPlot(ABC):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_redisp", test=True, _add_to_name=type(self).__name__)
        self.obs = self.env.current_obs
        self.plot = self._plotter(self.env.observation_space)

    def tearDown(self):
        self.env.close()

    @abstractmethod
    def _plotter(self, obs_space):
        pass

    def test_plot_layout(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_layout()

    def test_plot_info_line(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_info(
                line_values=self.obs.rho, gen_values=None, load_values=None
            )

    def test_plot_info_gen(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_info(
                line_values=None, gen_values=self.obs.prod_q, load_values=None
            )

    def test_plot_info_load(self):
        self.plot.plot_info(
            line_values=None, gen_values=None, load_values=self.obs.load_q
        )

    def test_plot_obs_default(self):
        self.plot.plot_obs(self.obs)

    def test_plot_obs_volts(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info="v", gen_info="v", load_info="v")

    def test_plot_obs_invalid_line(self):
        with self.assertRaises(PlotError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.plot.plot_obs(
                    self.obs, line_info="error", gen_info="v", load_info="v"
                )

    def test_plot_obs_no_line(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info=None, gen_info="p", load_info="p")

    def test_plot_obs_invalid_gen(self):
        with self.assertRaises(PlotError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.plot.plot_obs(
                    self.obs, line_info="v", gen_info="error", load_info="v"
                )

    def test_plot_obs_no_gen(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info="rho", gen_info=None, load_info="p")

    def test_plot_obs_invalid_load(self):
        with self.assertRaises(PlotError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.plot.plot_obs(
                    self.obs, line_info="rho", gen_info="v", load_info="error"
                )

    def test_plot_obs_no_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info="v", gen_info="v", load_info=None)

    def test_plot_obs_line(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info="a", gen_info=None, load_info=None)

    def test_plot_obs_gen(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info=None, gen_info="p", load_info=None)

    def test_plot_obs_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.plot.plot_obs(self.obs, line_info=None, gen_info=None, load_info="p")


class TestPlotMatplot(BaseTestPlot, unittest.TestCase):
    def _plotter(self, obs_space):
        return PlotMatplot(obs_space)


class TestPlotPlotly(BaseTestPlot, unittest.TestCase):
    def _plotter(self, obs_space):
        return PlotPlotly(obs_space)


if __name__ == "__main__":
    unittest.main()
