# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

import numpy as np

from grid2op.dtypes import dt_float

from .dispatchTypes import CurtailmentResult, RedispatchState


class CurtailmentModule:
    """Applies the curtailment limits to the renewable generators and computes the power removed."""
    def __init__(self, env) -> None:
        self.env = env

    def reset(self, state: RedispatchState) -> None:
        state.reset("curtailment", masks={"gen_renewable": self.env.gen_renewable})

    def _update_env_action(self, new_p) -> None:
        if "prod_p" in self.env._env_modification._dict_inj:
            self.env._env_modification._dict_inj["prod_p"][:] = new_p
        else:
            self.env._env_modification._dict_inj["prod_p"] = 1.0 * new_p
            self.env._env_modification._modif_inj = True

    def _update_limits(self, action, state: RedispatchState) -> None:
        curtailment_act = 1.0 * action._curtail
        ind_curtailed_in_act = (curtailment_act != -1.0) & self.env.gen_renewable
        state.limit_curtailment_prev[:] = state.limit_curtailment
        state.limit_curtailment[ind_curtailed_in_act] = curtailment_act[
            ind_curtailed_in_act
        ]

    def _apply_limits(self, new_p, curtailment_vect):
        gen_curtailed = np.abs(curtailment_vect - 1.0) >= 1e-7
        max_action = self.env.gen_pmax[gen_curtailed] * curtailment_vect[gen_curtailed]
        new_p[gen_curtailed] = np.minimum(max_action, new_p[gen_curtailed])
        return gen_curtailed

    def compute(self, action, new_p, state: RedispatchState) -> CurtailmentResult:
        if self.env.redispatching_unit_commitment_availble and (
            action._modif_curtailment
            or (np.abs(state.limit_curtailment - 1.0) >= 1e-7).any()
        ):
            self._update_limits(action, state)
            gen_curtailed = self._apply_limits(new_p, state.limit_curtailment)
            tmp_sum_curtailment_mw = dt_float(
                new_p[gen_curtailed].sum()
                - state.gen_before_curtailment[gen_curtailed].sum()
            )
            state.sum_curtailment_mw = (
                tmp_sum_curtailment_mw - state.sum_curtailment_mw_prev
            )
            state.sum_curtailment_mw_prev = tmp_sum_curtailment_mw
            self._update_env_action(new_p)
        else:
            state.sum_curtailment_mw = -state.sum_curtailment_mw_prev
            state.sum_curtailment_mw_prev = dt_float(0.0)
            gen_curtailed = np.abs(state.limit_curtailment - 1.0) >= 1e-7

        return CurtailmentResult(
            sum_curtailment_mw=state.sum_curtailment_mw,
            gen_curtailed=gen_curtailed,
        )
