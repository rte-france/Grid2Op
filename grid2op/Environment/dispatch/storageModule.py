# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

import numpy as np

from .dispatchTypes import RedispatchState, StorageResult


class StorageModule:
    """Computes the charge of the storage units and the power they absorb from the grid."""
    def __init__(self, env) -> None:
        self.env = env

    def reset(self, state: RedispatchState) -> None:
        if self.env.n_storage > 0:
            tmp = self.env._parameters.INIT_STORAGE_CAPACITY * self.env.storage_Emax
            if self.env._parameters.ACTIVATE_STORAGE_LOSS:
                tmp += self.env.storage_loss * self.env.delta_time_seconds / 3600.0
            state.reset("storage")
            state.storage_previous_charge[:] = tmp
            state.storage_current_charge[:] = tmp

    def withdraw_losses(self, state: RedispatchState) -> None:
        if self.env._parameters.ACTIVATE_STORAGE_LOSS:
            tmp_ = self.env.storage_loss * self.env.delta_time_seconds / 3600.0
            state.storage_current_charge -= tmp_
            state.storage_current_charge[:] = np.maximum(
                state.storage_current_charge, 0.0
            )

    def _clamp_too_high(self, delta_, indx_too_high, state: RedispatchState) -> None:
        coeff_p_to_E = self.env.delta_time_seconds / 3600.0
        tmp_ = 1.0 / coeff_p_to_E * delta_
        if self.env._parameters.ACTIVATE_STORAGE_LOSS:
            tmp_ /= self.env.storage_charging_efficiency[indx_too_high]
        state.storage_power[indx_too_high] -= tmp_

    def _clamp_too_low(self, delta_, indx_too_low, state: RedispatchState) -> None:
        coeff_p_to_E = self.env.delta_time_seconds / 3600.0
        tmp_ = 1.0 / coeff_p_to_E * delta_
        if self.env._parameters.ACTIVATE_STORAGE_LOSS:
            tmp_ *= self.env.storage_discharging_efficiency[indx_too_low]
        state.storage_power[indx_too_low] -= tmp_

    def compute(self, action_storage_power, state: RedispatchState) -> StorageResult:
        state.storage_previous_charge[:] = state.storage_current_charge
        storage_act = np.isfinite(action_storage_power) & (
            np.abs(action_storage_power) >= 1e-7
        )
        state.action_storage[:] = 0.0
        state.storage_power[:] = 0.0
        modif = False
        coeff_p_to_E = self.env.delta_time_seconds / 3600.0
        if storage_act.any():
            modif = True
            this_act_stor = action_storage_power[storage_act]
            eff_ = np.ones(storage_act.sum())
            if self.env._parameters.ACTIVATE_STORAGE_LOSS:
                fill_storage = this_act_stor > 0.0
                unfill_storage = this_act_stor < 0.0
                eff_[fill_storage] *= self.env.storage_charging_efficiency[storage_act][
                    fill_storage
                ]
                eff_[unfill_storage] /= self.env.storage_discharging_efficiency[
                    storage_act
                ][unfill_storage]
            state.storage_current_charge[storage_act] += this_act_stor * coeff_p_to_E * eff_
            state.action_storage[storage_act] += action_storage_power[storage_act]
            state.storage_power[storage_act] = this_act_stor

        if modif:
            indx_too_high = state.storage_current_charge > self.env.storage_Emax
            if indx_too_high.any():
                delta_ = (
                    state.storage_current_charge[indx_too_high]
                    - self.env.storage_Emax[indx_too_high]
                )
                self._clamp_too_high(delta_, indx_too_high, state)
                state.storage_current_charge[indx_too_high] = self.env.storage_Emax[
                    indx_too_high
                ]

            indx_too_low = state.storage_current_charge < self.env.storage_Emin
            if indx_too_low.any():
                delta_ = (
                    state.storage_current_charge[indx_too_low]
                    - self.env.storage_Emin[indx_too_low]
                )
                self._clamp_too_low(delta_, indx_too_low, state)
                state.storage_current_charge[indx_too_low] = self.env.storage_Emin[
                    indx_too_low
                ]

            state.storage_current_charge[:] = np.maximum(
                state.storage_current_charge, self.env.storage_Emin
            )
            state.amount_storage = state.storage_power.sum()
        else:
            state.amount_storage = 0.0

        tmp = state.amount_storage
        state.amount_storage -= state.amount_storage_prev
        state.amount_storage_prev = tmp
        self.withdraw_losses(state)
        return StorageResult(
            amount_storage_mw=state.amount_storage,
            storage_power=state.storage_power,
        )
