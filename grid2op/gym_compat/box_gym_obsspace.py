# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import numpy as np
from gym.spaces import Box

from grid2op.dtypes import dt_int, dt_bool, dt_float

# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part


class BoxGymObsSpace(Box):
    """
    TODO

    """
    def __init__(self,
                 grid2op_observation_space,
                 attr_to_keep=("gen_p", "load_p", "topo_vect"),
                 substract={},
                 divide={},
                 functs={}):
        self._attr_to_keep = attr_to_keep

        ob_sp = grid2op_observation_space
        self.dict_properties = {
            "year": (0, 2200, (1,), dt_int),
            "month": (0, 12, (1,), dt_int),
            "day": (0, 31, (1,), dt_int),
            "hour_of_day": (0, 24, (1,), dt_int),
            "minute_of_hour": (0, 60, (1,), dt_int),
            "day_of_week": (0, 7, (1,), dt_int),
            "gen_p": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                      1.2 * ob_sp.gen_pmax,
                      (ob_sp.n_gen,),
                      dt_float),
            "gen_q": (np.full(shape=(ob_sp.n_gen,), fill_value=-np.inf, dtype=dt_float),
                      np.full(shape=(ob_sp.n_gen,), fill_value=np.inf, dtype=dt_float),
                      (ob_sp.n_gen,),
                      dt_float),
            "gen_v": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                      np.full(shape=(ob_sp.n_gen,), fill_value=np.inf, dtype=dt_float),
                      (ob_sp.n_gen,),
                      dt_float),
            "load_p": (np.full(shape=(ob_sp.n_load,), fill_value=-np.inf, dtype=dt_float),
                       np.full(shape=(ob_sp.n_load,), fill_value=+np.inf, dtype=dt_float),
                       (ob_sp.n_load,),
                       dt_float),
            "load_q": (np.full(shape=(ob_sp.n_load,), fill_value=-np.inf, dtype=dt_float),
                       np.full(shape=(ob_sp.n_load,), fill_value=+np.inf, dtype=dt_float),
                       (ob_sp.n_load,),
                       dt_float),
            "laod_v": (np.full(shape=(ob_sp.n_load,), fill_value=0., dtype=dt_float),
                       np.full(shape=(ob_sp.n_load,), fill_value=np.inf, dtype=dt_float),
                       (ob_sp.n_load,),
                       dt_float),
            "p_or": (np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "q_or": (np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "a_or": (np.full(shape=(ob_sp.n_line,), fill_value=0., dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "v_or": (np.full(shape=(ob_sp.n_line,), fill_value=0., dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "p_ex": (np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "q_ex": (np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "a_ex": (np.full(shape=(ob_sp.n_line,), fill_value=0., dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "v_ex": (np.full(shape=(ob_sp.n_line,), fill_value=0., dtype=dt_float),
                     np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                     (ob_sp.n_line,),
                     dt_float),
            "rho": (np.full(shape=(ob_sp.n_line,), fill_value=0., dtype=dt_float),
                    np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_float),
                    (ob_sp.n_line,),
                    dt_float),
            "line_status": (np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                            np.full(shape=(ob_sp.n_line,), fill_value=1, dtype=dt_int),
                            (ob_sp.n_line,),
                            dt_int),
            "timestep_overflow": (np.full(shape=(ob_sp.n_line,), fill_value=-np.inf, dtype=dt_int),
                                  np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_int),
                                  (ob_sp.n_line,),
                                  dt_int),
            "topo_vect": (np.full(shape=(ob_sp.dim_topo,), fill_value=-1, dtype=dt_int),
                          np.full(shape=(ob_sp.dim_topo,), fill_value=2, dtype=dt_int),
                          (ob_sp.dim_topo,),
                          dt_int),
            "time_before_cooldown_line": (np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                                          np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_int),
                                          (ob_sp.n_line,),
                                          dt_int),
            "time_before_cooldown_sub": (np.full(shape=(ob_sp.n_sub,), fill_value=0, dtype=dt_int),
                                         np.full(shape=(ob_sp.n_sub,), fill_value=np.inf, dtype=dt_int),
                                         (ob_sp.n_sub,),
                                         dt_int),
            "time_next_maintenance": (np.full(shape=(ob_sp.n_line,), fill_value=-1, dtype=dt_int),
                                      np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_int),
                                      (ob_sp.n_line,),
                                      dt_int),
            "duration_next_maintenance": (np.full(shape=(ob_sp.n_line,), fill_value=0, dtype=dt_int),
                                          np.full(shape=(ob_sp.n_line,), fill_value=np.inf, dtype=dt_int),
                                          (ob_sp.n_line,),
                                          dt_int),
            "target_dispatch": (np.minimum(ob_sp.gen_pmin, -ob_sp.gen_pmax),
                                np.maximum(-ob_sp.gen_pmin, +ob_sp.gen_pmax),
                                (ob_sp.n_gen,),
                                dt_float),
            "actual_dispatch": (np.minimum(ob_sp.gen_pmin, -ob_sp.gen_pmax),
                                np.maximum(-ob_sp.gen_pmin, +ob_sp.gen_pmax),
                                (ob_sp.n_gen,),
                                dt_float),
            "storage_charge": (np.full(shape=(ob_sp.n_storage,), fill_value=0, dtype=dt_float),
                               1.0 * ob_sp.storage_Emax,
                               (ob_sp.n_storage,),
                               dt_float),
            "storage_power_target": (-1.0 * ob_sp.storage_max_p_prod,
                                     1.0 * ob_sp.storage_max_p_absorb,
                                     (ob_sp.n_storage,),
                                     dt_float),
            "storage_power": (-1.0 * ob_sp.storage_max_p_prod,
                              1.0 * ob_sp.storage_max_p_absorb,
                              (ob_sp.n_storage,),
                              dt_float),
            "curtailment": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                            np.full(shape=(ob_sp.n_gen,), fill_value=1.0, dtype=dt_float),
                            (ob_sp.n_gen,),
                            dt_float),
            "curtailment_limit": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                                  np.full(shape=(ob_sp.n_gen,), fill_value=1.0, dtype=dt_float),
                                  (ob_sp.n_gen,),
                                  dt_float),
            "curtailment_mw": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                               1.0 * ob_sp.gen_pmax,
                               (ob_sp.n_gen,),
                               dt_float),
            "curtailment_limit_mw": (np.full(shape=(ob_sp.n_gen,), fill_value=0., dtype=dt_float),
                                     1.0 * ob_sp.gen_pmax,
                                     (ob_sp.n_gen,),
                                     dt_float)
        }
        self.dict_properties["prod_p"] = self.dict_properties["gen_p"]
        self.dict_properties["prod_q"] = self.dict_properties["gen_q"]
        self.dict_properties["prod_v"] = self.dict_properties["gen_v"]

        self._substract = substract
        self._divide = divide

        # handle the "functional" part
        self._template_obs = ob_sp._template_obj.copy()
        self.__func = {}

        self._dims = None
        low, high, shape, dtype = self._get_info(functs)

        # initialize the base container
        Box.__init__(self, low=low, high=high, shape=shape, dtype=dtype)

    def _get_info(self, functs):
        low = None
        high = None
        shape = None
        dtype = None
        self._dims = []
        for el in self._attr_to_keep:
            if el in functs:
                # the attribute name "el" has been put in the functs
                callable_, low_, high_, shape_, dtype_ = functs[el]
                try:
                    tmp = callable_(self._template_obs.copy())
                except Exception as exc_:
                    raise RuntimeError(f"Error for the function your provided with key \"{el}\". "
                                       f"The error was :\n {exc_}")
                self.__func[el] = callable_
                if dtype_ is None:
                    dtype_ = dt_float
                if shape_ is None:
                    shape_ = tmp.shape
                if low_ is None:
                    low_ = np.full(shape_, fill_value=-np.inf, dtype=dtype_)
                elif isinstance(low_, float):
                    low_ = np.full(shape_, fill_value=low_, dtype=dtype_)

                if high_ is None:
                    high_ = np.full(shape_, fill_value=np.inf, dtype=dtype_)
                elif isinstance(high_, float):
                    high_ = np.full(shape_, fill_value=high_, dtype=dtype_)

            elif el in self.dict_properties:
                # el is an attribute of an observation, for example "load_q" or "topo_vect"
                low_, high_, shape_, dtype_ = self.dict_properties[el]
            else:
                li_keys = '\n\t-'.join(sorted(list(self.dict_properties.keys()) +
                                              list(self.__func.keys()))
                                       )
                raise RuntimeError(f"Unknown observation attributes \"{el}\". Supported attributes are: "
                                   f"\n{li_keys}")

            # handle the data type
            if dtype is None:
                dtype = dtype_
            else:
                if dtype_ == dt_float:
                    dtype = dt_float

            # handle the shape
            if shape is None:
                shape = shape_
            else:
                shape = (shape[0] + shape_[0],)

            # handle low / high
            if el in self._substract:
                low_ -= self._substract[el]
                high_ -= self._substract[el]
            if el in self._divide:
                low_ /= self._divide[el]
                high_ /= self._divide[el]
            if low is None:
                low = low_
                high = high_
            else:
                low = np.concatenate((low.astype(dtype), low_.astype(dtype))).astype(dtype)
                high = np.concatenate((high.astype(dtype), high_.astype(dtype))).astype(dtype)

            # remember where this need to be stored
            self._dims.append(shape[0])

        return low, high, shape, dtype

    def _handle_attribute(self, grid2op_observation, attr_nm):
        res = getattr(grid2op_observation, attr_nm).astype(self.dtype)
        if attr_nm in self._substract:
            res -= self._substract[attr_nm]
        if attr_nm in self._divide:
            res /= self._divide[attr_nm]
        return res

    def to_gym(self, grid2op_observation):
        """
        TODO
        Parameters
        ----------
        grid2op_observation

        Returns
        -------

        """
        res = np.empty(shape=self.shape, dtype=self.dtype)
        prev = 0
        for attr_nm, where_to_put in zip(self._attr_to_keep, self._dims):
            if attr_nm in self.__func:
                tmp = self.__func[attr_nm](grid2op_observation)
            elif hasattr(grid2op_observation, attr_nm):
                tmp = self._handle_attribute(grid2op_observation, attr_nm)
            else:
                raise RuntimeError(f"Unknown attribute \"{attr_nm}\".")
            res[prev:where_to_put] = tmp
            prev = where_to_put
        return res
