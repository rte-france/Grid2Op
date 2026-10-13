# Copyright (c) 2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from grid2op.Exceptions import ImpossibleRedispatching

from .baseRedispatchSolver import BaseRedispatchSolver
from .dispatchTypes import RedispatchConstraints, RedispatchResult, RedispatchState


class _ScaledSolverInput(NamedTuple):
    #: generators (among the participating ones) that appear in the objective
    already_modified_gen_me: np.ndarray
    #: target of these generators, divided by `scale_x`
    target_vals_me_optim: np.ndarray
    #: weight of each participating generator in the objective
    weights: np.ndarray
    #: scaling of the decision variables
    scale_x: float
    #: scaling of the objective function
    scale_objective: float


class DefaultRedispatchSolver(BaseRedispatchSolver):
    """The default solver of grid2op.

    It finds the dispatch closest (weighted least squares, the weights being inversely
    proportional to the ramps of the generators) to the target dispatch asked by the agent,
    under the constraints that the sum of the dispatch compensates the storage units,
    the curtailment and the detached elements and that pmin / pmax and the ramps are met.

    The problem is solved with the SLSQP method of scipy.
    """
    # to be compliant with scipy 1.16 (removes np.float32)
    _dt_float = float

    def _scale_solver_input(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
        gen_participating: np.ndarray,
    ) -> _ScaledSolverInput:
        this_dt_float = self._dt_float
        # define the objective value
        target_vals = (
            state.target_dispatch[gen_participating]
            - state.actual_dispatch[gen_participating]
        )
        already_modified_gen_me = state.already_modified_gen[gen_participating]
        target_vals_me = target_vals[already_modified_gen_me]
        nb_dispatchable = gen_participating.sum()
        coeffs = 1.0 / (constraints.ramp_up + constraints.ramp_down + constraints.epsilon)
        weights = np.ones(nb_dispatchable) * coeffs[gen_participating]
        weights /= weights.sum()

        if target_vals_me.shape[0] == 0:
            # no dispatch means all dispatchable, otherwise i will never get to 0
            already_modified_gen_me[:] = True
            target_vals_me = target_vals[already_modified_gen_me]

        # for numeric stability
        # to scale the input also:
        # see https://stackoverflow.com/questions/11155721/positive-directional-derivative-for-linesearch
        scale_x = max(np.max(np.abs(state.actual_dispatch)), 1.0)
        scale_x = this_dt_float(scale_x)
        target_vals_me_optim = 1.0 * (target_vals_me / scale_x)
        target_vals_me_optim = target_vals_me_optim.astype(this_dt_float)

        # see https://stackoverflow.com/questions/11155721/positive-directional-derivative-for-linesearch
        # where they advised to scale the function
        scale_objective = max(0.5 * np.abs(target_vals_me_optim).sum() ** 2, 1.0)
        scale_objective = np.round(scale_objective, decimals=4)
        scale_objective = this_dt_float(scale_objective)
        return _ScaledSolverInput(
            already_modified_gen_me=already_modified_gen_me,
            target_vals_me_optim=target_vals_me_optim,
            weights=weights,
            scale_x=scale_x,
            scale_objective=scale_objective,
        )

    def solve(
        self,
        constraints: RedispatchConstraints,
        state: RedispatchState,
    ) -> RedispatchResult:
        gen_participating, incr_in_chronics, failure = self._prepare_solver_inputs(
            constraints, state
        )
        if failure is not None:
            return failure

        this_dt_float = self._dt_float
        scaled = self._scale_solver_input(constraints, state, gen_participating)
        already_modified_gen_me = scaled.already_modified_gen_me
        target_vals_me_optim = scaled.target_vals_me_optim
        weights = scaled.weights
        scale_x = scaled.scale_x
        scale_objective = scaled.scale_objective
        nb_dispatchable = gen_participating.sum()
        tmp_zeros = np.zeros((1, nb_dispatchable), dtype=this_dt_float)
        new_p = constraints.new_p

        # add the "sum to 0"
        mat_sum_0_no_turn_on = np.ones((1, nb_dispatchable), dtype=this_dt_float)
        # this is where the storage units, the curtailment, the detached elements... are
        # taken into account: the dispatch must produce what they add or remove
        const_sum_0_no_turn_on = (
            np.zeros(1, dtype=this_dt_float)
            + constraints.power_to_compensate_mw
        )

        # gen increase in the chronics
        new_p_th = new_p[gen_participating] + state.actual_dispatch[gen_participating]

        # minimum value available for disp
        ## first limit delta because of pmin
        p_min_const = constraints.pmin[gen_participating] - new_p_th
        ## second limit delta because of ramps
        ramp_down_const = (
            -constraints.ramp_down[gen_participating]
            - incr_in_chronics[gen_participating]
        )
        ## take max of the 2
        min_disp = np.maximum(p_min_const, ramp_down_const)
        min_disp = min_disp.astype(this_dt_float)

        # maximum value available for disp
        ## first limit delta because of pmax
        p_max_const = constraints.pmax[gen_participating] - new_p_th
        ## second limit delta because of ramps
        ramp_up_const = (
            constraints.ramp_up[gen_participating]
            - incr_in_chronics[gen_participating]
        )
        ## take min of the 2
        max_disp = np.minimum(p_max_const, ramp_up_const)
        max_disp = max_disp.astype(this_dt_float)

        # add everything into a linear constraint object
        # equality
        added = 0.5 * constraints.epsilon
        equality_const = LinearConstraint(
            mat_sum_0_no_turn_on,  # do the sum
            const_sum_0_no_turn_on / scale_x,  # lower bound
            const_sum_0_no_turn_on / scale_x,  # upper bound
        )
        mat_pmin_max_ramps = np.eye(nb_dispatchable)
        ineq_const = LinearConstraint(
            mat_pmin_max_ramps,
            (min_disp - added) / scale_x,
            (max_disp + added) / scale_x,
        )

        # choose a good initial point (close to the solution)
        # the idea here is to chose a initial point that would be close to the
        # desired solution (split the (sum of the) dispatch to the available generators)
        x0 = np.zeros(nb_dispatchable, dtype=this_dt_float)
        if (np.abs(state.target_dispatch) >= 1e-7).any() or state.already_modified_gen.any():
            gen_for_x0 = np.abs(state.target_dispatch[gen_participating]) >= 1e-7
            gen_for_x0 |= state.already_modified_gen[gen_participating]
            x0[gen_for_x0] = (
                state.target_dispatch[gen_participating][gen_for_x0]
                - state.actual_dispatch[gen_participating][gen_for_x0]
            ) / scale_x
            # at this point x0 is made of the difference between the target and the
            # actual dispatch for all generators that have a
            # target dispatch non 0.

            # in this "if" block I set the other component of x0 to
            # their "right" value
            can_adjust = np.abs(x0) <= 1e-7
            if can_adjust.any():
                init_sum = x0.sum()
                denom_adjust = (1.0 / weights[can_adjust]).sum()
                if denom_adjust <= 1e-2:
                    # i don't want to divide by something too close to 0.
                    denom_adjust = 1.0
                x0[can_adjust] = -init_sum / (weights[can_adjust] * denom_adjust)
        else:
            # to "force" the exact reset to 0.0 for all components
            x0 -= state.actual_dispatch[gen_participating] / scale_x

        def target(actual_dispatchable):
            # define my real objective
            quad_ = (
                actual_dispatchable[already_modified_gen_me] - target_vals_me_optim
            ) ** 2
            coeffs_quads = weights[already_modified_gen_me] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            coeffs_quads_const /= scale_objective  # scaling the function
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res_jac = 1.0 * tmp_zeros
            res_jac[0, already_modified_gen_me] = (
                2.0
                * weights[already_modified_gen_me]
                * (actual_dispatchable[already_modified_gen_me] - target_vals_me_optim)
            )
            res_jac /= scale_objective  # scaling the function
            return res_jac.reshape(-1)

        res = minimize(
            target,
            x0,
            method="SLSQP",
            constraints=[equality_const, ineq_const],
            options={
                "eps": max(this_dt_float(constraints.epsilon / scale_x), 1e-6),
                "ftol": max(this_dt_float(constraints.epsilon / scale_x), 1e-6),
                "disp": False,
            },
            jac=jac,
        )
        new_dispatch = state.actual_dispatch.copy()
        if res.success:
            new_dispatch[gen_participating] += res.x * scale_x
            return RedispatchResult(success=True, actual_dispatch=new_dispatch)

        # check if constraints are "approximately" met
        mat_const = np.concatenate((mat_sum_0_no_turn_on, mat_pmin_max_ramps))
        downs = np.concatenate(
            (const_sum_0_no_turn_on / scale_x, (min_disp - added) / scale_x)
        )
        ups = np.concatenate(
            (const_sum_0_no_turn_on / scale_x, (max_disp + added) / scale_x)
        )
        vals = np.matmul(mat_const, res.x)
        ok_down = np.all(vals - downs >= -constraints.tol)  # i don't violate "down" constraints
        ok_up = np.all(vals - ups <= constraints.tol)
        if ok_up and ok_down:
            # it's ok i can tolerate "small" perturbations
            new_dispatch[gen_participating] += res.x * scale_x
            return RedispatchResult(success=True, actual_dispatch=new_dispatch)

        error_dispatch = (
            "Redispatching automaton terminated with error (no more information available "
            'at this point):\n"{}"'.format(res.message)
        )
        # the optimizer does not tell how much could not be dispatched
        return RedispatchResult.failed(ImpossibleRedispatching(error_dispatch))
