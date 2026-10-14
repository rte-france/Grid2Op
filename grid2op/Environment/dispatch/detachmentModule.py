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

from .dispatchTypes import DetachmentResult, RedispatchState


class DetachmentModule:
    """Computes the power of the loads and generators detached by the agent, that the other generators have to compensate.

    All the data of the step are given explicitly to :func:`DetachmentModule.compute`, this
    module does not read nor modify the attributes of the environment.
    """
    def __init__(self, env) -> None:
        self.env = env

    def compute(
        self,
        new_p: np.ndarray,
        new_p_th: np.ndarray,
        new_load_p: np.ndarray,
        state: RedispatchState,
        backend_action,
    ) -> DetachmentResult:
        """
        Parameters
        ----------
        new_p:
            Active production setpoint of the generators for this step (modified in place:
            the detached generators are set to 0.)
        new_p_th:
            Same as `new_p` before curtailment (modified in place as well)
        new_load_p:
            Active consumption of the loads for this step, as given by the time series (the
            injections of the agent are not taken into account). It is only read.
        state:
            The dispatch state of the environment, updated in place
        backend_action:
            The backend action of this step, it tells which elements are detached
        """
        gen_detached_user = backend_action.get_gen_detached()
        load_detached_user = backend_action.get_load_detached()

        mw_gen_lost_this = new_p[gen_detached_user].sum()
        mw_load_lost_this = new_load_p[load_detached_user].sum()
        total_power_lost = -mw_gen_lost_this + mw_load_lost_this
        state.detached_elements_mw = dt_float(
            -total_power_lost
            + state.actual_dispatch[gen_detached_user].sum()
            - state.detached_elements_mw_prev
        )
        state.detached_elements_mw_prev = dt_float(-total_power_lost)

        new_p[gen_detached_user] = 0.0
        new_p_th[gen_detached_user] = 0.0
        state.actual_dispatch[gen_detached_user] = 0.0
        return DetachmentResult(
            detached_mw=state.detached_elements_mw,
            gen_detached=gen_detached_user,
            new_p=new_p,
            new_p_th=new_p_th,
        )
