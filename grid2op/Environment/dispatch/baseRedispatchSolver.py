# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from grid2op.Exceptions import ImpossibleRedispatching

from .dispatchTypes import RedispatchConstraints, RedispatchResult, RedispatchState

DETAILED_REDISP_ERR_MSG = (
    "\nThis is an attempt to explain why the dispatch did not succeed and caused a game over.\n"
    "To compensate the {increase} of loads and / or {decrease} of "
    "renewable energy (due to naturl causes but also through curtailment) and / or variation in the storage units, "
    "the generators should {increase} their total production of {sum_move:.2f}MW (in total).\n"
    "But, if you take into account the generator constraints ({pmax} and {max_ramp_up}) you "
    "can have at most {avail_up_sum:.2f}MW.\n"
    "Indeed at time t, generators are in state:\n\t{gen_setpoint}\ntheir ramp max is:"
    "\n\t{ramp_up}\n and pmax is:\n\t{gen_pmax}\n"
    "Wrapping up, each generator can {increase} at {maximum} of:\n\t{avail_up}\n"
    "NB: if you did not do any dispatch during this episode, it would have been possible to "
    "meet these constraints. This situation is caused by not having enough degree of freedom "
    'to "compensate" the variation of the load due to (most likely) an "over usage" of '
    "redispatching feature (some generators stuck at {pmax} as a consequence of your "
    "redispatching. They can't increase their productions to meet the {increase} in demand or "
    "{decrease} of renewables)"
)


class BaseRedispatchSolver(ABC):
    """Base class of the redispatch solvers.

    A solver computes, at each step, the new ``actual_dispatch`` of the generators so that
    they produce ``constraints.power_to_compensate_mw`` more than the time series (to
    compensate the storage units, the curtailment, the detached elements...) while pmin /
    pmax and the ramps are respected.

    Only :func:`BaseRedispatchSolver.solve` has to be implemented. Everything else
    (tracking the target dispatch, checking that a redispatching action is valid, the
    minimum up / down times of the generators) is handled by the environment.

    A solver is copied for every environment that uses it (the environment itself, the one
    used by `obs.simulate`, the ones built by the runner, a copy of the environment...), so
    it must be copyable with :func:`copy.deepcopy` and must not share state between
    environments.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Environment.dispatch import BaseRedispatchSolver, RedispatchResult

        class ProportionalSolver(BaseRedispatchSolver):
            def solve(self, constraints, state):
                new_dispatch = 1.0 * state.actual_dispatch
                # ... compute the new dispatch
                return RedispatchResult(success=True, actual_dispatch=new_dispatch)
                # or, if no dispatch can be found:
                # return RedispatchResult.failed(ImpossibleRedispatching("..."), unserved_mw=...)

        env = grid2op.make("l2rpn_case14_sandbox", redispatch_solver=ProportionalSolver)

    """
    def __init__(self) -> None:
        #: the environment using this solver, set by :func:`BaseRedispatchSolver.bind`
        self.env = None

    def bind(self, env) -> "BaseRedispatchSolver":
        """Attach the solver to the environment that uses it."""
        self.env = env
        return self

    def copy_for_env(self, env) -> "BaseRedispatchSolver":
        """Return a deep copy of this solver bound to `env` (which can be ``None``)."""
        # a subclass might not call `BaseRedispatchSolver.__init__`
        env_ = getattr(self, "env", None)
        self.env = None  # the environment is not copied with the solver
        try:
            new_obj = copy.deepcopy(self)
        finally:
            self.env = env_
        return new_obj.bind(env)

    @abstractmethod
    def solve(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
    ) -> RedispatchResult:
        """Compute the new dispatch.

        It must not modify `state` (nor `constraints`): the environment applies the
        dispatch of the result if it succeeded.

        Returns
        -------
        The :class:`RedispatchResult`. If it did not succeed, its exception causes a game
        over and its `unserved_mw` tells how much power the generators could not compensate.
        """

    def reset(self) -> None:
        """Called when the environment is reset. Override it if the solver has an internal
        state (by default it does nothing)."""
        pass

    def _prepare_solver_inputs(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[RedispatchResult]]:
        """Compute the generators taking part in the dispatch and how much the time
        series make them move, and check that the problem is feasible.

        Returns
        -------
        gen_participating:
            Mask of the generators that the solver can modify
        incr_in_chronics:
            Variation of the production of each generator due to the time series
        failure:
            ``None`` if the problem is feasible, otherwise the (failed) result to return
        """
        new_p = constraints.new_p
        gen_participating = constraints.gen_participating.copy()
        incr_in_chronics = new_p - (state.gen_activeprod_t_redisp - state.actual_dispatch)

        # check if the constraints are violated
        failure = self._check_feasibility(constraints, state, gen_participating, incr_in_chronics)
        if failure is None:
            return gen_participating, incr_in_chronics, None
        if not constraints.can_use_all_redispatchable:
            return gen_participating, incr_in_chronics, failure

        # try to force the turn on of turned off generators (if parameters allow it)
        gen_participating_tmp = constraints.redispatchable_mask.copy()
        gen_participating_tmp[constraints.gen_detached] = False
        failure_tmp = self._check_feasibility(constraints, state, gen_participating_tmp, incr_in_chronics)
        if failure_tmp is not None:
            return gen_participating, incr_in_chronics, failure_tmp
        # I can "save" the situation by turning on all generators, I do it
        return gen_participating_tmp, incr_in_chronics, None

    def _check_feasibility(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
        gen_participating: np.ndarray,
        incr_in_chronics: np.ndarray,
    ) -> Optional[RedispatchResult]:
        ## total available "juice" to go down (incl ramp and pmin / pmax)
        p_min_down = (
            constraints.pmin[gen_participating]
            - state.gen_activeprod_t_redisp[gen_participating]
        )
        avail_down = np.maximum(p_min_down, -constraints.ramp_down[gen_participating])
        ## total available "juice" to go up (incl. ramp and pmin / pmax)
        p_max_up = (
            constraints.pmax[gen_participating]
            - state.gen_activeprod_t_redisp[gen_participating]
        )
        avail_up = np.minimum(p_max_up, constraints.ramp_up[gen_participating])
        return self._detect_infeasible_dispatch(
            constraints, incr_in_chronics[gen_participating], avail_down, avail_up, state
        )

    def _detect_infeasible_dispatch(
        self,
        constraints: RedispatchConstraints,
        incr_in_chronics: np.ndarray,
        avail_down: np.ndarray,
        avail_up: np.ndarray,
        state: RedispatchState,
    ) -> Optional[RedispatchResult]:
        """This function is an attempt to give more detailed log by detecting infeasible dispatch"""
        # what the generators have to produce more (time series + storage, curtailment...)
        sum_move = incr_in_chronics.sum() + constraints.power_to_compensate_mw
        avail_down_sum = avail_down.sum()
        avail_up_sum = avail_up.sum()
        redisp = constraints.redispatchable_mask
        gen_setpoint = state.gen_activeprod_t_redisp[redisp]
        if sum_move > avail_up_sum:
            # infeasible because too much is asked
            msg = DETAILED_REDISP_ERR_MSG.format(
                sum_move=sum_move,
                avail_up_sum=avail_up_sum,
                gen_setpoint=np.round(gen_setpoint, decimals=2),
                ramp_up=constraints.ramp_up[redisp],
                gen_pmax=constraints.pmax[redisp],
                avail_up=np.round(avail_up, decimals=2),
                increase="increase",
                decrease="decrease",
                maximum="maximum",
                pmax="pmax",
                max_ramp_up="max_ramp_up",
            )
            return RedispatchResult.failed(ImpossibleRedispatching(msg),
                                           unserved_mw=float(sum_move - avail_up_sum))
        elif sum_move < avail_down_sum:
            # infeasible because not enough is asked
            msg = DETAILED_REDISP_ERR_MSG.format(
                sum_move=sum_move,
                avail_up_sum=avail_down_sum,
                gen_setpoint=np.round(gen_setpoint, decimals=2),
                ramp_up=constraints.ramp_down[redisp],
                gen_pmax=constraints.pmin[redisp],
                avail_up=np.round(avail_up, decimals=2),
                increase="decrease",
                decrease="increase",
                maximum="minimum",
                pmax="pmin",
                max_ramp_up="max_ramp_down",
            )
            return RedispatchResult.failed(ImpossibleRedispatching(msg),
                                           unserved_mw=float(sum_move - avail_down_sum))
        return None
