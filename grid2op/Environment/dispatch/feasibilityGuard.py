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

from .dispatchTypes import GuardInfo, RedispatchState


class FeasibilityGuard:
    """Limits the curtailment and the storage actions when the generators cannot
    compensate them (not enough ramps or margin to pmin / pmax), so that a dispatch can
    still be found. Used when ``LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION`` is set."""
    def __init__(self, env) -> None:
        self.env = env

    def _readjust_curtailment(
        self,
        total_curtailment,
        new_p_th,
        new_p,
        state: RedispatchState,
    ) -> None:
        state.sum_curtailment_mw += total_curtailment
        state.sum_curtailment_mw_prev += total_curtailment
        if total_curtailment > self.env._tol_poly:
            curtailed = new_p_th - new_p
        else:
            new_p_with_previous_curtailment = 1.0 * new_p_th
            self.env._curtailment_module._apply_limits(
                new_p_with_previous_curtailment, state.limit_curtailment_prev
            )
            curtailed = new_p_th - new_p_with_previous_curtailment

        curt_sum = curtailed.sum()
        if abs(curt_sum) > self.env._tol_poly:
            curtailed[~self.env.gen_renewable] = 0.0
            curtailed *= total_curtailment / curt_sum
            new_p[self.env.gen_renewable] += curtailed[self.env.gen_renewable]

    def _readjust_storage(self, total_storage, state: RedispatchState) -> None:
        new_act_storage = 1.0 * state.storage_power
        sum_this_step = new_act_storage.sum()
        if (abs(sum_this_step) > self.env._tol_poly
            and abs(total_storage) <= abs(sum_this_step) + self.env._tol_poly):
            # (this includes cancelling the current action completely)
            modif_storage = new_act_storage * total_storage / sum_this_step
        else:
            new_act_storage = 1.0 * state.storage_power_prev
            sum_this_step = new_act_storage.sum()
            if abs(sum_this_step) > 1e-1:
                modif_storage = new_act_storage * total_storage / sum_this_step
            else:
                modif_storage = new_act_storage

        coeff_p_to_E = self.env.delta_time_seconds / 3600.0
        state.storage_power -= modif_storage

        # the efficiency depends on the direction of the resulting storage power, or on the
        # one of the cancelled action when the action is cancelled completely
        direction = 1.0 * state.storage_power
        cancelled = np.abs(direction) <= 1e-7
        direction[cancelled] = modif_storage[cancelled]
        is_discharging = direction < 0.0
        is_charging = direction > 0.0
        modif_storage[is_discharging] /= type(self.env).storage_discharging_efficiency[
            is_discharging
        ]
        modif_storage[is_charging] *= type(self.env).storage_charging_efficiency[
            is_charging
        ]

        state.storage_current_charge -= coeff_p_to_E * modif_storage
        state.amount_storage -= total_storage
        state.amount_storage_prev -= total_storage

    @staticmethod
    def _clamp_to_adjustable(too_much, total_storage_curtail):
        """Storage and curtailment can at most be cancelled, not reversed: what they
        cannot absorb is left to the solver (which then reports the infeasibility)."""
        if np.sign(too_much) != np.sign(total_storage_curtail):
            # limiting them would make the dispatch even harder
            return dt_float(0.0)
        if abs(too_much) > abs(total_storage_curtail):
            return dt_float(total_storage_curtail)
        return too_much

    def check_and_clamp(
        self,
        new_p,
        new_p_th,
        state: RedispatchState,
    ) -> GuardInfo:
        cls = type(self.env)
        # same generators as the ones the solver can use: the detached ones cannot move
        gen_redisp = cls.gen_redispatchable.copy()
        if cls.detachment_is_allowed:
            gen_redisp[self.env._backend_action.get_gen_detached()] = False
        normal_increase = new_p - (
            state.gen_activeprod_t_redisp - state.actual_dispatch
        )
        normal_increase = normal_increase[gen_redisp]
        p_min_down = self.env.gen_pmin[gen_redisp] - state.gen_activeprod_t_redisp[gen_redisp]
        avail_down = np.maximum(p_min_down, -self.env.gen_max_ramp_down[gen_redisp])
        p_max_up = self.env.gen_pmax[gen_redisp] - state.gen_activeprod_t_redisp[gen_redisp]
        avail_up = np.minimum(p_max_up, self.env.gen_max_ramp_up[gen_redisp])

        # the power of the detached elements also has to be compensated by the generators
        # (as in the solver) but it cannot be limited here, only storage and curtailment can
        sum_move = (
            normal_increase.sum()
            + state.amount_storage
            - state.sum_curtailment_mw
            + state.detached_elements_mw
        )
        total_storage_curtail = state.amount_storage - state.sum_curtailment_mw
        update_env_act = False
        total_curtailment = dt_float(0.0)
        total_storage = dt_float(0.0)

        if abs(total_storage_curtail) >= self.env._tol_poly:
            too_much = 0.0
            if sum_move > avail_up.sum():
                too_much = dt_float(sum_move - avail_up.sum() + self.env._tol_poly)
                too_much = self._clamp_to_adjustable(too_much, total_storage_curtail)
                state.limited_before = too_much
            elif sum_move < avail_down.sum():
                too_much = dt_float(sum_move - avail_down.sum() - self.env._tol_poly)
                too_much = self._clamp_to_adjustable(too_much, total_storage_curtail)
                state.limited_before = too_much
            elif np.abs(state.limited_before) >= self.env._tol_poly:
                update_env_act = True
                too_much = min(avail_up.sum() - self.env._tol_poly, state.limited_before)
                state.limited_before -= too_much
                too_much = state.limited_before

            if abs(too_much) > self.env._tol_poly:
                total_curtailment = dt_float(
                    -state.sum_curtailment_mw / total_storage_curtail * too_much
                )
                total_storage = dt_float(
                    state.amount_storage / total_storage_curtail * too_much
                )
                update_env_act = True

                if np.sign(total_curtailment) != np.sign(total_storage):
                    total_curtailment = (
                        too_much
                        if np.sign(total_curtailment) == np.sign(too_much)
                        else 0.0
                    )
                    total_storage = (
                        too_much if np.sign(total_storage) == np.sign(too_much) else 0.0
                    )

                self._readjust_curtailment(total_curtailment, new_p_th, new_p, state)
                self._readjust_storage(total_storage, state)

            if update_env_act:
                self.env._curtailment_module._update_env_action(new_p)

        return GuardInfo(
            total_curtailment=dt_float(total_curtailment),
            total_storage=dt_float(total_storage),
            updated_env_action=update_env_act,
        )
