# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from dataclasses import dataclass, field, fields
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


#: kind of a :class:`RedispatchState` field: one value per generator, storage unit or load,
#: or a single value
DISPATCH_FIELD_KINDS = ("gen", "storage", "load", "scalar")

#: groups of :class:`RedispatchState` fields, reset together by :func:`RedispatchState.reset`
DISPATCH_FIELD_GROUPS = ("redispatch", "storage", "curtailment", "detachment")


def _state_field(
    kind: str,
    dtype: type,
    init,
    group: str,
    *,
    restore: bool = True,
    reset: bool = True,
    reset_mask: Optional[str] = None,
):
    """Declares a field of :class:`RedispatchState`.

    Parameters
    ----------
    kind:
        One of :data:`DISPATCH_FIELD_KINDS`, gives the size of the field
    dtype:
        ``dt_float``, ``dt_int`` or ``dt_bool``
    init:
        Value of the field (of all its entries for a vector) once allocated or reset
    group:
        One of :data:`DISPATCH_FIELD_GROUPS`
    restore:
        Whether the field is restored from an observation (used by `obs.simulate` and
        `obs.get_forecast_env()`)
    reset:
        Whether :func:`RedispatchState.reset` resets the field
    reset_mask:
        Name of a boolean mask (given to :func:`RedispatchState.reset`): only the entries
        of this mask are reset
    """
    if kind not in DISPATCH_FIELD_KINDS:
        raise ValueError(f"Unknown kind {kind}")
    if group not in DISPATCH_FIELD_GROUPS:
        raise ValueError(f"Unknown group {group}")
    metadata = {
        "kind": kind,
        "dtype": dtype,
        "init": init,
        "group": group,
        "restore": restore,
        "reset": reset,
        "reset_mask": reset_mask,
    }
    default = dtype(init) if kind == "scalar" else None
    return field(default=default, metadata=metadata)


@dataclass
class RedispatchState:
    """State of the generators, storage units and curtailment kept by the environment
    from one step to the next. The environment owns it, the solver only updates
    ``actual_dispatch``.

    Each field is declared once, with :func:`_state_field`: allocation, copy, reset,
    the conversion to / from a dictionary (to restore the state from an observation) and
    the ``BaseEnv._<field>`` attributes are derived from these declarations.
    """
    # redispatch
    target_dispatch: Optional[np.ndarray] = _state_field("gen", dt_float, 0.0, "redispatch")
    already_modified_gen: Optional[np.ndarray] = _state_field("gen", dt_bool, False, "redispatch")
    actual_dispatch: Optional[np.ndarray] = _state_field("gen", dt_float, 0.0, "redispatch")
    gen_uptime: Optional[np.ndarray] = _state_field("gen", dt_int, 0, "redispatch")
    gen_downtime: Optional[np.ndarray] = _state_field("gen", dt_int, 0, "redispatch")
    gen_activeprod_t: Optional[np.ndarray] = _state_field("gen", dt_float, 0.0, "redispatch")
    gen_activeprod_t_redisp: Optional[np.ndarray] = _state_field("gen", dt_float, 0.0, "redispatch")

    # storage units
    storage_current_charge: Optional[np.ndarray] = _state_field("storage", dt_float, 0.0, "storage")
    storage_previous_charge: Optional[np.ndarray] = _state_field("storage", dt_float, 0.0, "storage")
    # not reset: it is set to 0. at the beginning of each step
    action_storage: Optional[np.ndarray] = _state_field("storage", dt_float, 0.0, "storage", reset=False)
    amount_storage: float = _state_field("scalar", dt_float, 0.0, "storage")
    amount_storage_prev: float = _state_field("scalar", dt_float, 0.0, "storage")
    storage_power: Optional[np.ndarray] = _state_field("storage", dt_float, 0.0, "storage")
    storage_power_prev: Optional[np.ndarray] = _state_field("storage", dt_float, 0.0, "storage")

    # curtailment (the limits are reset only for the renewable generators)
    limit_curtailment: Optional[np.ndarray] = _state_field(
        "gen", dt_float, 1.0, "curtailment", reset_mask="gen_renewable")
    limit_curtailment_prev: Optional[np.ndarray] = _state_field(
        "gen", dt_float, 1.0, "curtailment", reset_mask="gen_renewable")
    gen_before_curtailment: Optional[np.ndarray] = _state_field("gen", dt_float, 0.0, "curtailment")
    sum_curtailment_mw: float = _state_field("scalar", dt_float, 0.0, "curtailment")
    sum_curtailment_mw_prev: float = _state_field("scalar", dt_float, 0.0, "curtailment")
    limited_before: float = _state_field("scalar", dt_float, 0.0, "curtailment")

    # detachment
    detached_elements_mw: float = _state_field("scalar", dt_float, 0.0, "detachment")
    detached_elements_mw_prev: float = _state_field("scalar", dt_float, 0.0, "detachment")

    @classmethod
    def allocate(cls, n_gen: int, n_storage: int, n_load: int) -> "RedispatchState":
        sizes = {"gen": n_gen, "storage": n_storage, "load": n_load}
        res = cls()
        for fld in fields(cls):
            meta = fld.metadata
            if meta["kind"] != "scalar":
                setattr(res, fld.name,
                        np.full(sizes[meta["kind"]], fill_value=meta["init"], dtype=meta["dtype"]))
        return res

    def copy(self) -> "RedispatchState":
        """Copy of this state, that does not share any array with it."""
        res = type(self)()
        for fld in fields(self):
            val = getattr(self, fld.name)
            if isinstance(val, np.ndarray):
                val = val.copy()
            setattr(res, fld.name, val)
        return res

    def reset(
        self,
        group: Optional[str] = None,
        masks: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """Sets the fields of `group` (all the fields if ``None``) to their initial value.

        `masks` gives the masks named by the `reset_mask` of the fields (only the entries
        of the mask are reset); it is required for these fields.
        """
        if group is not None and group not in DISPATCH_FIELD_GROUPS:
            raise ValueError(f"Unknown group {group}")
        for fld in fields(self):
            meta = fld.metadata
            if not meta["reset"] or (group is not None and meta["group"] != group):
                continue
            if meta["kind"] == "scalar":
                setattr(self, fld.name, meta["dtype"](meta["init"]))
                continue
            arr = getattr(self, fld.name)
            if arr is None:
                # not allocated
                continue
            if meta["reset_mask"] is None:
                arr[:] = meta["init"]
            else:
                if masks is None or meta["reset_mask"] not in masks:
                    raise KeyError(f"The mask \"{meta['reset_mask']}\" is needed to reset {fld.name}")
                arr[masks[meta["reset_mask"]]] = meta["init"]

    def to_dict(self) -> Dict[str, object]:
        """Copy of the values of all the fields (to restore them with :func:`RedispatchState.load_from`)."""
        res = {}
        for fld in fields(self):
            val = getattr(self, fld.name)
            res[fld.name] = val.copy() if isinstance(val, np.ndarray) else val
        return res

    def load_from(self, dict_: Mapping[str, object]) -> None:
        """Sets the fields declared with ``restore=True`` from `dict_` (as given by
        :func:`RedispatchState.to_dict`). The arrays are modified in place."""
        for fld in fields(self):
            meta = fld.metadata
            if not meta["restore"]:
                continue
            val = dict_[fld.name]
            if meta["kind"] == "scalar":
                setattr(self, fld.name, meta["dtype"](val))
            else:
                getattr(self, fld.name)[:] = val


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
