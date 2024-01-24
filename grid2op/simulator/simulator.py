# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
from typing import Optional, Tuple
import numpy as np
import os
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from grid2op.dtypes import dt_float
from grid2op.Environment import BaseEnv
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Observation.highresSimCounter import HighResSimCounter
from grid2op.Exceptions import SimulatorError, InvalidRedispatching


class Simulator(object):
    """This class represents a "simulator". It allows to check the impact on this or that on th powergrid, quite 
    like what human operators have at their disposal in control rooms.
    
    It behaves similarly to `env.step(...)` or `obs.simulate(...)` with a few key differences:
    
    - you can "chain" the call to simulator: `simulator.predict(...).predict(...).predict(...)`
    - it does not take into account the "time": no cooldown on lines nor substation, storage
      "state of charge" (energy) does not decrease when you use them
    - no automatic line disconnection: lines are not disconnected when they are above their limit
    - no opponent will act on the grid
    
    Please see the documentation for usage examples.
    
    """
    def __init__(
        self, backend: Optional[Backend], env: Optional[BaseEnv] = None, tol_redisp=1e-6,
        _highres_sim_counter: Optional[HighResSimCounter] =None
    ):
        # backend should be initiliazed !
        if backend is not None:
            if not isinstance(backend, Backend):
                raise SimulatorError(
                    f'The "backend" argument should be an object '
                    f'of type "Backend" you provided {backend}'
                )
            if env is not None:
                raise SimulatorError(
                    "When building a simulator with a grid2op backend "
                    'make sure you set the kwarg "env=None"'
                )
            if backend._can_be_copied:
                self.backend: Backend = backend.copy()
            else:
                raise SimulatorError("Impossible to make a Simulator when you "
                                     "cannot copy the backend.")
        else:
            if env is None:
                raise SimulatorError(
                    "If you want to build a simulator with a blank / None "
                    'backend you should provide an environment (kwargs "env")'
                )
            if not isinstance(env, BaseEnv):
                raise SimulatorError(
                    f"Make sure the environment you provided is "
                    f"a grid2op Environment (an object of a type "
                    f"inheriting from BaseEnv"
                )
            if env.backend._can_be_copied:
                self.backend = env.backend.copy()
            else:
                raise SimulatorError("Impossible to make a Simulator when you "
                                     "cannot copy the backend of the environment.")

        self.current_obs: BaseObservation = None
        self._converged: Optional[bool] = None
        self._error: Optional[Exception] = None

        self._tol_redisp: float = tol_redisp
        if _highres_sim_counter is not None:
            self._highres_sim_counter = _highres_sim_counter
        else:
            self._highres_sim_counter = HighResSimCounter()

    @property
    def converged(self) -> bool:
        """
        
        Returns
        -------
        bool
            Whether or not the powerflow has converged
        """
        return self._converged

    @converged.setter
    def converged(self, values):
        raise SimulatorError("Cannot set this property.")

    def copy(self) -> "Simulator":
        """Allows to perform a (deep) copy of the simulator.

        Returns
        -------
        Simulator
            A (deep) copy of the simulator you want to copy.

        Raises
        ------
        SimulatorError
            In case the simulator is not initialized.
            
        """
        if self.current_obs is None:
            raise SimulatorError(
                "Impossible to copy a non initialized Simulator. "
                "Have you used `simulator.set_state(obs, ...)` with a valid observation before ?"
            )
        res = copy.copy(self)
        res.backend = res.backend.copy()
        res.current_obs = res.current_obs.copy()
        # do not copy this !
        res._highres_sim_counter = self._highres_sim_counter
        return res

    def change_backend(self, backend: Backend):
        """You can use this function in case you want to change the "solver" use to perform the computation.
        
        For example, you could use a machine learning based model to do the computation (to accelerate them), provided
        that you have at your disposal such an algorithm. 

        .. warning::
            The backend you pass as argument should be initialized with the same grid as the one currently in use.
        
        Notes
        -----
        Once changed, all the "simulator" that "derived" from this simulator will use the same backend types.
        
        Parameters
        ----------
        backend : Backend
            Another grid2op backend you can use to perform the computation.

        Raises
        ------
        SimulatorError
            When you do not pass a correct backend.
            
        """
        if not isinstance(backend, Backend):
            raise SimulatorError(
                "when using change_backend function, the backend should"
                " be an object (an not a class) of type backend"
            )
        self.backend.close()
        self.backend = backend.copy()  # backend_class.init_grid(type(self.backend))
        self.set_state(obs=self.current_obs)

    def change_backend_type(self, backend_type: type, grid_path: os.PathLike, **kwargs):
        """It allows to change the type of the backend used

        Parameters
        ----------
        backend_type : type
            The new backend type
        grid_path : os.PathLike
            The path from where to load the powergrid
        kwargs:
            Extra arguments used to build the backend.
            
        Notes
        -----
        Once changed, all the "simulator" that "derived" from this simulator will use the same backend types.
        
        Raises
        ------
        SimulatorError
            if something went wrong (eg you do not pass a type, your type does not inherit from Backend, the file
            located at `grid_path` does not exists etc.)
        """
        if not isinstance(backend_type, type):
            raise SimulatorError(
                "when using change_backend_type function, the backend_type should"
                " be a class an not an object"
            )
        if not issubclass(backend_type, Backend):
            raise SimulatorError(
                "when using change_backend_type function, the backend_type should"
                " be subtype of class Backend"
            )
        if not os.path.exists(grid_path):
            raise SimulatorError(
                f'the supposed grid path "{grid_path}" does not exists'
            )
        if not os.path.isfile(grid_path):
            raise SimulatorError(f'the supposed grid path "{grid_path}" if not a file')

        if backend_type._IS_INIT:
            backend_type_init = backend_type
        else:
            backend_type_init= backend_type.init_grid(type(self.backend))
            
        tmp_backend = backend_type_init(**kwargs)
        # load a forecasted grid if there are any
        path_env, grid_name_with_ext = os.path.split(grid_path)
        grid_name, ext = os.path.splitext(grid_name_with_ext)
        grid_forecast_name = f"{grid_name}_forecast.{ext}"
        if os.path.exists(os.path.join(path_env, grid_forecast_name)):
            grid_path_loaded = os.path.join(path_env, grid_forecast_name)
        else:
            grid_path_loaded = grid_path 
        tmp_backend.load_grid(grid_path_loaded)
        tmp_backend.assert_grid_correct()
        tmp_backend.runpf()
        tmp_backend.assert_grid_correct_after_powerflow()
        tmp_backend.set_thermal_limit(self.backend.get_thermal_limit())
        self.backend.close()
        self.backend = tmp_backend
        self.set_state(obs=self.current_obs)

    def set_state(
        self,
        obs: Optional[BaseObservation] = None,
        do_powerflow: bool = True,
        new_gen_p: np.ndarray = None,
        new_gen_v: np.ndarray = None,
        new_load_p: np.ndarray = None,
        new_load_q: np.ndarray = None,
        update_thermal_limit: bool = True,
    ):
        """Set the state of the simulator to a given state described by an observation (and optionally some
        new loads and generation)

        Parameters
        ----------
        obs : Optional[BaseObservation], optional
            The observation to get the state from, by default None
        do_powerflow : bool, optional
            Whether to use the underlying backend to get a consistent state after
            this modification or not, by default True
        new_gen_p : np.ndarray, optional
            new generator active setpoint, by default None
        new_gen_v : np.ndarray, optional
            new generator voltage setpoint, by default None
        new_load_p : np.ndarray, optional
            new load active consumption, by default None
        new_load_q : np.ndarray, optional
            new load reactive consumption, by default None
        update_thermal_limit: bool, optional
            Do you update the thermal limit of the backend (we recommend to leave it to `True`
            otherwise some bugs can appear such as 
            https://github.com/rte-france/Grid2Op/issues/377)

        Raises
        ------
        SimulatorError
            In case the current simulator is not initialized.
        """

        if obs is not None:
            self.current_obs = obs.copy()

        if self.current_obs is None:
            raise SimulatorError(
                "The simulator is not initialized. Have you used `simulator.set_state(obs, ...)` with a valid observation before ?"
            )

        # you cannot use "simulate" of the observation in this class
        self.current_obs._obs_env = None
        self.current_obs._forecasted_inj = []
        self.current_obs._forecasted_grid = []

        # udpate the new state if needed
        if new_load_p is not None:
            self.current_obs.load_p[:] = new_load_p
        if new_load_q is not None:
            self.current_obs.load_q[:] = new_load_q

        if new_gen_p is not None:
            self.current_obs.gen_p[:] = new_gen_p
        if new_gen_v is not None:
            self.current_obs.gen_v[:] = new_gen_v

        self._converged = None
        self.error = None
        self.backend.update_from_obs(self.current_obs, force_update=True)

        if update_thermal_limit:
            self.backend.update_thermal_limit_from_vect(self.current_obs.thermal_limit)
            
        if do_powerflow:
            self._do_powerflow()

    def _do_powerflow(self):
        self._highres_sim_counter.add_one()
        self._converged, self._error = self.backend.runpf()

    def _update_obs(self):
        if self._converged:
            self.current_obs._update_attr_backend(self.backend)
        else:
            self.current_obs.set_game_over()

    def _adjust_controlable_gen(
        self, new_gen_p: np.ndarray, target_dispatch: np.ndarray, sum_target: float
    ) -> Optional[float]:
        nb_dispatchable = self.current_obs.gen_redispatchable.sum()

        # which generators needs to be "optimized" -> the one where
        # the target function matter
        gen_in_target = target_dispatch[self.current_obs.gen_redispatchable] != 0.0

        # compute the upper / lower bounds for the generators
        dispatchable = new_gen_p[self.current_obs.gen_redispatchable]
        val_min = (
            self.current_obs.gen_pmin[self.current_obs.gen_redispatchable]
            - dispatchable
        )
        val_max = (
            self.current_obs.gen_pmax[self.current_obs.gen_redispatchable]
            - dispatchable
        )

        # define the target function (things that will be minimized)
        target_dispatch_redisp = target_dispatch[self.current_obs.gen_redispatchable]

        coeffs = 1.0 / (
            self.current_obs.gen_max_ramp_up
            + self.current_obs.gen_max_ramp_down
            + self._tol_redisp
        )
        weights = np.ones(nb_dispatchable) * coeffs[self.current_obs.gen_redispatchable]
        weights /= weights.sum()

        scale_objective = max(0.5 * np.abs(target_dispatch_redisp).sum() ** 2, 1.0)
        scale_objective = np.round(scale_objective, decimals=4)

        tmp_zeros = np.zeros((1, nb_dispatchable), dtype=float)
        
        # wrap everything into the proper scipy form
        def target(actual_dispatchable):
            # define my real objective
            quad_ = (
                1e2
                * (
                    actual_dispatchable[gen_in_target]
                    - target_dispatch_redisp[gen_in_target]
                )
                ** 2
            )
            coeffs_quads = weights[gen_in_target] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            coeffs_quads_const /= scale_objective  # scaling the function
            coeffs_quads_const += 1e-2 * (actual_dispatchable**2 * weights).sum()
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res_jac = 1.0 * tmp_zeros
            res_jac[0, gen_in_target] = (
                1e2
                * 2.0
                * weights[gen_in_target]
                * (
                    actual_dispatchable[gen_in_target]
                    - target_dispatch_redisp[gen_in_target]
                )
            )
            res_jac /= scale_objective  # scaling the function
            res_jac += 2e-2 * actual_dispatchable * weights
            return res_jac

        mat_sum_ok = np.ones((1, nb_dispatchable))
        equality_const = LinearConstraint(
            mat_sum_ok, sum_target - self._tol_redisp, sum_target + self._tol_redisp
        )

        ineq_const = LinearConstraint(np.eye(nb_dispatchable), lb=val_min, ub=val_max)
        # objective function
        def f(init):
            this_res = minimize(
                target,
                init,
                method="SLSQP",
                constraints=[equality_const, ineq_const],
                options={
                    "eps": self._tol_redisp,
                    "ftol": self._tol_redisp,
                    "disp": False,
                },
                jac=jac,
            )
            return this_res

        # choose a good initial point (close to the solution)
        # the idea here is to chose a initial point that would be close to the
        # desired solution (split the (sum of the) dispatch to the available generators)
        x0 = 1.0 * target_dispatch_redisp
        can_adjust = x0 == 0.0
        if (can_adjust).any():
            init_sum = x0.sum()
            denom_adjust = (1.0 / weights[can_adjust]).sum()
            if denom_adjust <= 1e-2:
                # i don't want to divide by something too cloose to 0.
                denom_adjust = 1.0
            x0[can_adjust] = -init_sum / (weights[can_adjust] * denom_adjust)

        res = f(x0.astype(float))
        if res.success:
            return res.x
        else:
            return None

    def _amount_curtailed(
        self, act: BaseAction, new_gen_p: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        curt_vect = 1.0 * act.curtail
        curt_vect[curt_vect == -1.0] = 1.0
        limit_curtail = curt_vect * act.gen_pmax
        curtailed = np.maximum(new_gen_p - limit_curtail, 0.0)
        curtailed[~act.gen_renewable] = 0.0
        amount_curtail = curtailed.sum()
        new_gen_p_after_curtail = 1.0 * new_gen_p
        new_gen_p_after_curtail -= curtailed
        return new_gen_p_after_curtail, amount_curtail

    def _amount_storage(self, act: BaseAction) -> Tuple[float, np.ndarray]:
        storage_act = 1.0 * act.storage_p
        res = self.current_obs.storage_power_target.sum()
        current_charge = 1.0 * self.current_obs.storage_charge
        storage_power = np.zeros(act.n_storage)
        if np.all(np.abs(storage_act) <= self._tol_redisp):
            return -res, storage_power, current_charge
        coeff_p_to_E = (
            self.current_obs.delta_time / 60.0
        )  # obs.delta_time is in minutes

        # convert power (action to energy)
        storage_act_E = storage_act * coeff_p_to_E
        # take into account the efficiencies
        do_charge = storage_act_E < 0.0
        do_discharge = storage_act_E > 0.0
        storage_act_E[do_charge] /= act.storage_charging_efficiency[do_charge]
        storage_act_E[do_discharge] *= act.storage_discharging_efficiency[do_discharge]
        # make sure we don't go over / above Emin / Emax
        min_down_E = act.storage_Emin - current_charge
        min_up_E = act.storage_Emax - current_charge
        storage_act_E = np.minimum(storage_act_E, min_up_E)
        storage_act_E = np.maximum(storage_act_E, min_down_E)
        current_charge += storage_act_E
        # convert back to power (for the observation) the amount the grid got
        storage_power = storage_act_E / coeff_p_to_E
        storage_power[do_charge] *= act.storage_charging_efficiency[do_charge]
        storage_power[do_discharge] /= act.storage_discharging_efficiency[do_discharge]
        res += storage_power.sum()
        return -res, storage_power, current_charge

    def _fix_redisp_curtailment_storage(
        self, act: BaseAction, new_gen_p: np.ndarray
    ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """This function emulates the "frequency control" of the
        environment.

        Its main goal is to ensure that the sum of injected power thanks to redispatching,
        storage units and curtailment sum to 0.

        It is a very rough simplification of what happens in the environment.
        """
        new_gen_p_after_curtail, amount_curtail = self._amount_curtailed(act, new_gen_p)
        amount_storage, storage_power, storage_charge = self._amount_storage(act)
        sum_target = amount_curtail - amount_storage  # TODO !

        target_dispatch = self.current_obs.target_dispatch + act.redispatch
        # if previous setpoint was say -2 and at this step I redispatch of
        # say + 4 then the real setpoint should be +2 (and not +4)
        new_vect_redisp = (act.redispatch != 0.0) & (
            self.current_obs.target_dispatch == 0.0
        )
        target_dispatch[new_vect_redisp] += self.current_obs.actual_dispatch[
            new_vect_redisp
        ]

        if abs(target_dispatch.sum() - sum_target) >= self._tol_redisp:
            adjust = self._adjust_controlable_gen(
                new_gen_p_after_curtail, target_dispatch, sum_target
            )
            if adjust is None:
                return True, None, None, None, None, None
            else:
                return (
                    True,
                    new_gen_p_after_curtail,
                    target_dispatch,
                    adjust,
                    storage_power,
                    storage_charge,
                )
        return False, None, None, None, None, None

    def predict(
        self,
        act: BaseAction,
        new_gen_p: np.ndarray = None,
        new_gen_v: np.ndarray = None,
        new_load_p: np.ndarray = None,
        new_load_q: np.ndarray = None,
        do_copy: bool = True,
    ) -> "Simulator":
        """Predict the state of the grid after a given action has been taken.

        Parameters
        ----------
        act : BaseAction
            The action you want to take
        new_gen_p : np.ndarray, optional
            the new production active setpoint, by default None
        new_gen_v : np.ndarray, optional
            the new production voltage setpoint, by default None
        new_load_p : np.ndarray, optional
            the new consumption active values, by default None
        new_load_q : np.ndarray, optional
            the new consumption reactive values, by default None
        do_copy : bool, optional
            Whether to make a copy or not, by default True
        
        Examples
        ---------
        
        A possible example is:
        
        .. code-block:: python
        
            import grid2op
            env_name = "l2rpn_case14_sandbox"  # or any other name
            env = grid2op.make(env_name)

            obs = env.reset()

            #### later in the code, for example in an Agent:

            simulator = obs.get_simulator()

            load_p_stressed = obs.load_p * 1.05
            gen_p_stressed = obs.gen_p * 1.05
            do_nothing = env.action_space()
            simulator_stressed = simulator.predict(act=do_nothing,
                                                new_gen_p=gen_p_stressed,
                                                new_load_p=load_p_stressed)
            if not simulator_stressed.converged:
                # the solver fails to find a solution for this action
                # you are likely to run into trouble if you use that...
                ...  # do something
            obs_stressed = simulator_stressed.current_obs
            
        Returns
        -------
        Simulator
            The new simulator representing the grid state after the simulation of the action.
            
        """
        # init the result
        if do_copy:
            res = self.copy()
        else:
            res = self
        this_act = act.copy()

        if new_gen_p is None:
            new_gen_p = 1.0 * self.current_obs.gen_p

        res.set_state(
            obs=None,
            new_gen_p=new_gen_p,
            new_gen_v=new_gen_v,
            new_load_p=new_load_p,
            new_load_q=new_load_q,
            do_powerflow=False,
        )

        # "fix" the action for the redispatching / curtailment / storage part
        (
            has_adjusted,
            new_gen_p_modif,
            target_dispatch,
            adjust,
            storage_power,
            storage_charge,
        ) = res._fix_redisp_curtailment_storage(this_act, new_gen_p)

        if has_adjusted:
            if target_dispatch is None:
                res._converged = False
                res.current_obs.set_game_over()
                res._error = InvalidRedispatching("")
                return res

            redisp_modif = np.zeros(self.current_obs.n_gen)
            redisp_modif[self.current_obs.gen_redispatchable] = adjust
            # adjust the proper things in the observation
            res.current_obs.target_dispatch = target_dispatch
            this_act.redispatch = redisp_modif
            res.current_obs.actual_dispatch[:] = redisp_modif
            this_act._dict_inj["prod_p"] = 1.0 * new_gen_p_modif
            this_act._modif_inj = True

            # TODO : curtail, curtailment_limit (in observation)
            res.current_obs.curtailment[:] = (
                new_gen_p - new_gen_p_modif
            ) / act.gen_pmax
            res.current_obs.curtailment_limit[:] = act.curtail
            res.current_obs.curtailment_limit_effective[:] = act.curtail
            res.current_obs.gen_p_before_curtail[:] = new_gen_p
            res.current_obs.storage_power[:] = storage_power
            res.current_obs.storage_charge[:] = storage_charge
        else:
            res.current_obs.storage_power[:] = 0.0
            res.current_obs.actual_dispatch[:] = 0.0

        # apply the action
        bk_act = res.backend.my_bk_act_class()
        bk_act += this_act
        res.backend.apply_action(bk_act)

        # run the powerflow
        res._do_powerflow()

        # update its observation
        res._update_obs()
        return res

    def close(self):
        """close the underlying backend"""
        if hasattr(self, "backend") and self.backend is not None:
            self.backend.close()
        self.backend = None
        self.current_obs = None
        self._converged = None
        self._error = None

    def __del__(self):
        self.close()
