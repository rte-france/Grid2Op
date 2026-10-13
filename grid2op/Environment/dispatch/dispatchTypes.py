# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np

from grid2op.dtypes import dt_bool, dt_float, dt_int

if TYPE_CHECKING:
    from grid2op.Environment.baseEnv import BaseEnv


def _read_only(arr: np.ndarray) -> np.ndarray:
    """Read only view of `arr`: some of the arrays given to the solver are class attributes
    of the environment (shared by all the environments) or its working vectors."""
    res = arr.view()
    res.flags.writeable = False
    return res


@dataclass
class StorageResult:
    amount_storage_mw: float = dt_float(0.0)
    storage_power: Optional[np.ndarray] = None


@dataclass
class CurtailmentResult:
    sum_curtailment_mw: float = dt_float(0.0)
    gen_curtailed: Optional[np.ndarray] = None


@dataclass
class DetachmentResult:
    detached_mw: float = dt_float(0.0)
    gen_detached: Optional[np.ndarray] = None
    new_p: Optional[np.ndarray] = None
    new_p_th: Optional[np.ndarray] = None


@dataclass
class GuardInfo:
    total_curtailment: float = dt_float(0.0)
    total_storage: float = dt_float(0.0)
    updated_env_action: bool = False


@dataclass
class RedispatchState:
    """State of the generators, storage units and curtailment kept by the environment
    from one step to the next. The environment owns it, the solver only updates
    ``actual_dispatch``."""
    target_dispatch: Optional[np.ndarray] = None
    already_modified_gen: Optional[np.ndarray] = None
    actual_dispatch: Optional[np.ndarray] = None
    gen_uptime: Optional[np.ndarray] = None
    gen_downtime: Optional[np.ndarray] = None
    gen_activeprod_t: Optional[np.ndarray] = None
    gen_activeprod_t_redisp: Optional[np.ndarray] = None
    storage_current_charge: Optional[np.ndarray] = None
    storage_previous_charge: Optional[np.ndarray] = None
    action_storage: Optional[np.ndarray] = None
    amount_storage: float = dt_float(0.0)
    amount_storage_prev: float = dt_float(0.0)
    storage_power: Optional[np.ndarray] = None
    storage_power_prev: Optional[np.ndarray] = None
    limit_curtailment: Optional[np.ndarray] = None
    limit_curtailment_prev: Optional[np.ndarray] = None
    gen_before_curtailment: Optional[np.ndarray] = None
    sum_curtailment_mw: float = dt_float(0.0)
    sum_curtailment_mw_prev: float = dt_float(0.0)
    detached_elements_mw: float = dt_float(0.0)
    detached_elements_mw_prev: float = dt_float(0.0)
    limited_before: float = dt_float(0.0)

    @classmethod
    def allocate(cls, n_gen: int, n_storage: int) -> "RedispatchState":
        return cls(
            target_dispatch=np.zeros(n_gen, dtype=dt_float),
            already_modified_gen=np.zeros(n_gen, dtype=dt_bool),
            actual_dispatch=np.zeros(n_gen, dtype=dt_float),
            gen_uptime=np.zeros(n_gen, dtype=dt_int),
            gen_downtime=np.zeros(n_gen, dtype=dt_int),
            gen_activeprod_t=np.zeros(n_gen, dtype=dt_float),
            gen_activeprod_t_redisp=np.zeros(n_gen, dtype=dt_float),
            storage_current_charge=np.zeros(n_storage, dtype=dt_float),
            storage_previous_charge=np.zeros(n_storage, dtype=dt_float),
            action_storage=np.zeros(n_storage, dtype=dt_float),
            storage_power=np.zeros(n_storage, dtype=dt_float),
            storage_power_prev=np.zeros(n_storage, dtype=dt_float),
            limit_curtailment=np.ones(n_gen, dtype=dt_float),
            limit_curtailment_prev=np.ones(n_gen, dtype=dt_float),
            gen_before_curtailment=np.zeros(n_gen, dtype=dt_float),
        )


def dispatch_contributions(state: RedispatchState) -> Dict[str, float]:
    """Power (in MW) that the generators have to produce in addition to the time series,
    for each source. Positive means the generators must produce more.

    This is the only place where a new source (load shedding, deferred loads...) has to be
    declared for the dispatch to take it into account.
    """
    return {
        # storage units are in load convention: when they absorb, the generators produce more
        "storage": float(state.amount_storage),
        # curtailment removes renewable production: the other generators produce more
        "curtailment": -float(state.sum_curtailment_mw),
        # detached loads (resp. generators) make the other generators produce less (resp. more)
        "detachment": float(state.detached_elements_mw),
    }


def total_power_to_compensate(contributions: Mapping[str, float]) -> float:
    """Sum of the contributions (in MW)."""
    res = 0.0
    for val in contributions.values():
        res += val
    return res


@dataclass
class RedispatchResult:
    """What a redispatch solver returns.

    The solver does not modify the state of the environment: if the dispatch succeeds the
    environment sets its ``actual_dispatch`` to the one of the result.
    """
    #: whether a dispatch meeting all the constraints has been found
    success: bool
    #: the new dispatch of all the generators (in MW, one per generator), only when `success`
    actual_dispatch: Optional[np.ndarray] = None
    #: why no dispatch could be found, only when not `success`
    exception: Optional[Exception] = None
    #: the power (in MW, same sign convention as `power_to_compensate_mw`) that the generators
    #: cannot compensate. 0. on success, ``nan`` when the solver cannot tell.
    unserved_mw: float = 0.0

    @classmethod
    def failed(cls, exception: Exception, unserved_mw: float = float("nan")) -> "RedispatchResult":
        return cls(success=False, exception=exception, unserved_mw=unserved_mw)


@dataclass(frozen=True)
class RedispatchConstraints:
    """Everything a redispatch solver needs to know about the current step.

    It is built by the environment only when a dispatch has to be computed. The arrays
    have one entry per generator (``n_gen``), the powers are in MW. They are read only
    (copy them before modifying them).

    The solver must find the change of ``actual_dispatch`` for the participating generators
    such that its sum equals ``power_to_compensate_mw`` (the power the generators have to
    produce in addition to the time series because of the storage units, the curtailment,
    the detached elements...) while respecting pmin / pmax and the ramps.
    """
    #: productions (in MW) of the generators given by the time series, after curtailment and detachment
    new_p: np.ndarray
    #: generators that can be modified by the solver (redispatchable, not detached and
    #: either producing or already dispatched)
    gen_participating: np.ndarray
    #: generators detached by the agent this step (all ``False`` if detachment is not allowed)
    gen_detached: np.ndarray
    #: power (in MW) the generators have to produce in addition to the time series, for all
    #: the sources together (positive: the generators must produce more). This is the only
    #: value a solver needs about the balance, it is the sum of `contributions`
    power_to_compensate_mw: float
    #: the same power, split by source ("storage", "curtailment", "detachment"), with the same
    #: sign convention. For information only (error messages, logs...)
    contributions: Mapping[str, float]
    pmin: np.ndarray
    pmax: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    redispatchable_mask: np.ndarray
    #: whether the solver may make every redispatchable generator participate when the
    #: participating ones are not enough (``IGNORE_MIN_UP_DOWN_TIME`` and
    #: ``ALLOW_DISPATCH_GEN_SWITCH_OFF`` are both set)
    can_use_all_redispatchable: bool
    #: small value used to relax the bounds of the problem
    epsilon: float
    #: tolerance on the constraints
    tol: float

    @classmethod
    def from_state(
        cls,
        new_p: np.ndarray,
        state: RedispatchState,
        env: "BaseEnv",
    ) -> "RedispatchConstraints":
        cls_env = type(env)
        if cls_env.detachment_is_allowed:
            gen_detached = env._backend_action.get_gen_detached()
        else:
            gen_detached = np.zeros(cls_env.n_gen, dtype=dt_bool)
        # these are the generators that will be adjusted for redispatching
        gen_participating = (
            (new_p > 0.0)
            | (np.abs(state.actual_dispatch) >= 1e-7)
            | (state.target_dispatch != state.actual_dispatch)
        )
        gen_participating[~cls_env.gen_redispatchable] = False
        gen_participating[gen_detached] = False
        params = env._parameters
        contributions = dispatch_contributions(state)
        return cls(
            new_p=_read_only(new_p),
            gen_participating=_read_only(gen_participating),
            gen_detached=_read_only(gen_detached),
            power_to_compensate_mw=total_power_to_compensate(contributions),
            contributions=contributions,
            pmin=_read_only(cls_env.gen_pmin),
            pmax=_read_only(cls_env.gen_pmax),
            ramp_up=_read_only(cls_env.gen_max_ramp_up),
            ramp_down=_read_only(cls_env.gen_max_ramp_down),
            redispatchable_mask=_read_only(cls_env.gen_redispatchable),
            can_use_all_redispatchable=(
                params.IGNORE_MIN_UP_DOWN_TIME and params.ALLOW_DISPATCH_GEN_SWITCH_OFF
            ),
            epsilon=env._epsilon_poly,
            tol=env._tol_poly,
        )
