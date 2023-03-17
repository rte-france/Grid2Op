# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import pandapower as pp
import copy
import numpy as np

real_case = pp.from_json("grid.json")
sim_case = copy.deepcopy(real_case)

np.random.seed(42)
noise = 0.05

# change powerlines
sim_case.line["r_ohm_per_km"] *= np.random.lognormal(0., noise)
sim_case.line["x_ohm_per_km"] *= np.random.lognormal(0., noise)
# TODO do I change trafo ?

pp.runpp(sim_case)
pp.runpp(real_case)
assert sim_case.converged
assert sim_case.res_line.shape[0] == sim_case.line.shape[0]
print(f"L1 error on p: {np.mean(np.abs(sim_case.res_line['p_from_mw'] - real_case.res_line['p_from_mw'])):.2f}MW")
print(f"L1 error on q: {np.mean(np.abs(sim_case.res_line['q_from_mvar'] - real_case.res_line['q_from_mvar'])):.2f}MVAr")
pp.to_json(sim_case, "grid_forecast.json")
