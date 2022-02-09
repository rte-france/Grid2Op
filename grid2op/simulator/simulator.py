# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import copy
from typing import Optional
import numpy as np
import os
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from grid2op.Environment import BaseEnv
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Exceptions import SimulatorError, InvalidRedispatching

class Simulator(object):
    def __init__(self,
                 backend: Optional[Backend],
                 env : Optional[BaseEnv]=None,
                 tol_redisp=1e-6):
        # backend should be initiliazed !
        if backend is not None:
            if not isinstance(backend, Backend):
                raise SimulatorError(f"The \"backend\" argument should be an object "
                                     f"of type \"Backend\" you provided {backend}")
            if env is not None:
                raise SimulatorError("When building a simulator with a grid2op backend "
                                     "make sure you set the kwarg \"env=None\"")
            self.backend : Backend = backend.copy()
        else:
            if env is None:
                raise SimulatorError("If you want to build a simulator with a blank / None "
                                "backend you should provide an environment (kwargs \"env\")")
            if not isinstance(env, BaseEnv):
                raise SimulatorError(f"Make sure the environment you provided is "
                                     f"a grid2op Environment (an object of a type "
                                     f"inheriting from BaseEnv")
            self.backend = env.backend.copy()
            
        self.current_obs : BaseObservation = None
        self._converged : Optional[bool] = None
        self._error : Optional[Exception] = None
        
        self._tol_redisp : float= tol_redisp
    
    def copy(self) -> "Simulator":
        if self.current_obs is None:
            raise SimulatorError("Impossible to copy a non initialized Simulator. "
                                 "Have you used `simulator.set_state(obs, ...)` with a valid observation before ?")
        res = copy.copy(self)
        res.backend = res.backend.copy()
        res.current_obs = res.current_obs.copy()
        return res
    
    def change_backend(self, backend):
        if not isinstance(backend, Backend):
            raise SimulatorError("when using change_backend function, the backend should"
                                 " be an object (an not a class) of type backend")
        self.backend.close()
        self.backend = backend.copy()  # backend_class.init_grid(type(self.backend))
        self.set_state(obs=self.current_obs)
    
    def change_backend_type(self, backend_type, grid_path, **kwargs):
        if not isinstance(backend_type, type):
            raise SimulatorError("when using change_backend_type function, the backend_type should"
                                 " be a class an not an object")
        if not issubclass(backend_type, Backend):
            raise SimulatorError("when using change_backend_type function, the backend_type should"
                                 " be subtype of class Backend")
        if not os.path.exists(grid_path):
            raise Simulator(f"the supposed grid path \"{grid_path}\" does not exists")
        if not os.path.isfile(grid_path):
            raise Simulator(f"the supposed grid path \"{grid_path}\" if not a file")
        
        tmp_backend = backend_type(**kwargs)
        tmp_backend.load_grid(grid_path)
        tmp_backend.assert_grid_correct()
        self.backend.close()
        self.backend = tmp_backend
        self.set_state(obs=self.current_obs)
        
    def set_state(self,
                  obs: Optional[BaseObservation]=None,
                  do_powerflow: bool=True,
                  new_gen_p : np.array=None,
                  new_gen_v : np.array=None,
                  new_load_p : np.array=None,
                  new_load_q : np.array=None):
        
        if obs is not None:
            self.current_obs = obs.copy()
            
        if self.current_obs is None:
            raise SimulatorError("The simulator is not initialized. Have you used `simulator.set_state(obs, ...)` with a valid observation before ?")
      
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
            
        self.converged = None
        self.error = None
        self.backend.update_from_obs(self.current_obs, force_update=True)
        
        if do_powerflow:
            self._do_powerflow()
    
    def _do_powerflow(self):
        self._converged, self._error = self.backend.runpf()
            
    def _update_obs(self):
        if self._converged:
            self.current_obs._update_attr_backend(self.backend)
        else:
            self.current_obs.set_game_over()
        
    def _adjust_controlable_gen(self, new_gen_p, target_dispatch, sum_target):
        nb_dispatchable = np.sum(self.current_obs.gen_redispatchable)
        
        # which generators needs to be "optimized" -> the one where
        # the target function matter
        gen_in_target = target_dispatch[self.current_obs.gen_redispatchable] != 0.
        
        # compute the upper / lower bounds for the generators
        dispatchable = new_gen_p[self.current_obs.gen_redispatchable]
        val_min = self.current_obs.gen_pmin[self.current_obs.gen_redispatchable] - dispatchable
        val_max = self.current_obs.gen_pmax[self.current_obs.gen_redispatchable] - dispatchable
        
        # define the target function (things that will be minimized)
        target_dispatch_redisp = target_dispatch[self.current_obs.gen_redispatchable]
        
        coeffs = 1.0 / (self.current_obs.gen_max_ramp_up + self.current_obs.gen_max_ramp_down + self._tol_redisp)
        weights = np.ones(nb_dispatchable) * coeffs[self.current_obs.gen_redispatchable]
        weights /= weights.sum()
        
        scale_objective = max(0.5 * np.sum(np.abs(target_dispatch_redisp))**2, 1.0)
        scale_objective = np.round(scale_objective, decimals=4)
        scale_objective = scale_objective
        
        tmp_zeros = np.zeros((1, nb_dispatchable), dtype=float)
        
        # wrap everything into the proper scipy form
        def target(actual_dispatchable):
            # define my real objective
            quad_ = (actual_dispatchable[gen_in_target] - target_dispatch_redisp[gen_in_target]) ** 2
            coeffs_quads = weights[gen_in_target] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            coeffs_quads_const /= scale_objective  # scaling the function
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res_jac = 1.0 * tmp_zeros
            res_jac[0, gen_in_target] = 2.0 * weights[gen_in_target] * (actual_dispatchable[gen_in_target] - target_dispatch_redisp[gen_in_target])
            res_jac /= scale_objective  # scaling the function
            return res_jac
        
        mat_sum_0_no_turn_on = np.ones((1, nb_dispatchable))
        equality_const = LinearConstraint(mat_sum_0_no_turn_on,
                                          sum_target - self._tol_redisp,
                                          sum_target + self._tol_redisp)
        
        ineq_const = LinearConstraint(np.eye(nb_dispatchable),
                                      lb=val_min,
                                      ub=val_max)
        # objective function
        def f(init):
            this_res = minimize(target,
                                init,
                                method="SLSQP",
                                constraints=[equality_const, ineq_const],
                                options={'eps': self._tol_redisp,
                                         "ftol": self._tol_redisp,
                                         'disp': False},
                                jac=jac,
                                )
            return this_res
        
        # choose a good initial point (close to the solution)
        # the idea here is to chose a initial point that would be close to the
        # desired solution (split the (sum of the) dispatch to the available generators)
        x0 = 1.0 * target_dispatch_redisp
        can_adjust = x0 == 0.
        if np.any(can_adjust):
            init_sum = np.sum(x0)
            denom_adjust = np.sum(1. / weights[can_adjust])
            if denom_adjust <= 1e-2:
                # i don't want to divide by something too cloose to 0.
                denom_adjust = 1.0
            x0[can_adjust] = - init_sum / (weights[can_adjust] * denom_adjust)
            
        res = f(x0)
        if res.success:
            return res.x
        else:
            return None
            
    def _fix_redisp_curtailment_storage(self, act, new_gen_p):
        """This function emulates the "frequency control" of the 
        environment.
        
        Its main goal is to ensure that the sum of injected power thanks to redispatching,
        storage units and curtailment sum to 0.
        
        It is a very rough simplification of what happens in the environment.
        """
        
        sum_target = 0. # TODO !
        new_vect_redisp = (act.redispatch != 0.) & (self.current_obs.target_dispatch == 0.)
        target_dispatch = self.current_obs.target_dispatch + act.redispatch
        # if previous setpoint was say -2 and at this step I redispatch of
        # say + 4 then the real setpoint should be +2 (and not +4)
        target_dispatch[new_vect_redisp] += self.current_obs.actual_dispatch[new_vect_redisp]
            
        if (np.sum(target_dispatch) - sum_target) >= self._tol_redisp:
            if new_gen_p is None:
                new_gen_p = 1.0 * self.current_obs.gen_p
            adjust = self._adjust_controlable_gen(new_gen_p, target_dispatch, sum_target)
            if adjust is None:
                return True, None, None, None
            else:
                return True, new_gen_p, target_dispatch, adjust
        return False, None, None, None
        
    def predict(self,
                act: BaseAction,
                do_copy: bool=True,
                new_gen_p : np.array=None,
                new_gen_v : np.array=None,
                new_load_p : np.array=None,
                new_load_q : np.array=None) -> "Simulator":
        # init the result
        if do_copy:
            res = self.copy()
        else:
            res = self
        this_act = act.copy()
        
        res.set_state(obs=None, 
                      new_gen_p=new_gen_p,
                      new_gen_v=new_gen_v, 
                      new_load_p=new_load_p, 
                      new_load_q=new_load_q)
        
        # "fix" the action for the redispatching / curtailment / storage part
        has_adjusted, new_gen_p_modif, target_dispatch, adjust = res._fix_redisp_curtailment_storage(this_act, new_gen_p)
        
        if has_adjusted:
            if target_dispatch is None:
                res._converged = False
                res.current_obs.set_game_over()
                res._error = InvalidRedispatching("")
                return
            
            redisp_modif = np.zeros(self.current_obs.n_gen)
            redisp_modif[self.current_obs.gen_redispatchable] = adjust
            # adjust the proper things in the observation
            res.current_obs.target_dispatch = target_dispatch
            this_act.redispatch = redisp_modif
            res.current_obs.actual_dispatch = redisp_modif
            this_act._dict_inj["prod_p"] = 1.0 * new_gen_p_modif
            this_act._modif_inj = True
            # TODO : curtail, curtailment_limit, storage_power
            # TODO deactivate the storage state of charge !
        
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
        if self.backend is not None:
            self.backend.close()
        self.backend = None
        self.current_obs = None
        self._converged = None
        self._error = None
        
    def __del__(self):
        self.close()
        