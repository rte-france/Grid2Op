# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple
import copy
import warnings
import numpy as np

from grid2op.Action import BaseAction, ActionSpace
from grid2op.dtypes import dt_int, dt_bool, dt_float

from grid2op.Exceptions import Grid2OpException

# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part

from grid2op.gym_compat.utils import (ALL_ATTR_CONT,
                                      ATTR_DISCRETE,
                                      check_gym_version,
                                      GYM_AVAILABLE,
                                      GYMNASIUM_AVAILABLE)


class __AuxBoxGymActSpace:
    """
    This class allows to convert a grid2op action space into a gym "Box" which is
    a regular Box in R^d.

    It also allows to customize which part of the action you want to use and offer capacity to
    center / reduce the data or to use more complex function from the observation.

    .. note::
        Though it is possible to use every type of action with this type of action space, be aware that
        this is not recommended at all to use it for discrete attribute (set_bus, change_bus, set_line_status or
        change_line_status) !

        Basically, when doing action in gym for these attributes, this converter will involve rounding and
        is definitely not the best representation. Prefer the :class:`MultiDiscreteActSpace` or
        the :class:`DiscreteActSpace` classes.

    .. note::
        A gymnasium Box is encoded as a numpy array, see the example section for more information.
    
    .. danger::
        If you use this encoding for the "curtailment" you might end up with "weird" behaviour. Your agent
        will perfom some kind of curtailment at each step (there is no real way to express "I don't want to curtail")
        So the index corresponding to the "curtail" type of actions should rather be "1." (meaning
        "limit the value at 100%" which is somewhat equivalent to "I don't want to curtail")
        
    Examples
    --------
    If you simply want to use it you can do:

    .. code-block:: python

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv, BoxGymActSpace
        gym_env = GymEnv(env)

        gym_env.action_space = BoxGymActSpace(env.action_space)

    In this case it will extract all the features in all the action (a detailed list is given
    in the documentation at :ref:`action-module`).

    You can select the attribute you want to keep, for example:

    .. code-block:: python

        gym_env.action_space = BoxGymActSpace(env.action_space,
                                                   attr_to_keep=['redispatch', "curtail"])

    You can also apply some basic transformation to the attribute of the action. This can be done with:

    .. code-block:: python

        gym_env.action_space = BoxGymActSpace(env.action_space,
                                                   attr_to_keep=['redispatch', "curtail"],
                                                   multiply={"redispatch": env.gen_max_ramp_up},
                                                   add={"redispatch": 0.5 * env.gen_max_ramp_up})

    In the above example, the resulting "redispatch" part of the vector will be given by the following
    formula: `grid2op_act = gym_act * multiply + add`

    Hint: you can use: `multiply` being the standard deviation and `add` being the average of the attribute.

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`BoxGymActSpace` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`BoxGymnasiumActSpace`), otherwise it will
          inherit from gym (and will be exactly :class:`BoxLegacyGymActSpace`)
        - :class:`BoxGymnasiumActSpace` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`BoxLegacyGymActSpace` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    For the "l2rpn_case14_sandbox" environment, a code using :class:`BoxGymActSpace` can look something like
    (if you want to build action "by hands"):
    
    .. code-block:: python
    
        import grid2op
        from grid2op.gym_compat import GymEnv, BoxGymActSpace
        import numpy as np
        env_name = "l2rpn_case14_sandbox"
        
        env = grid2op.make(env_name)
        gym_env =  GymEnv(env)
        gym_env.action_space = BoxGymActSpace(env.action_space)
        
        obs = gym_env.reset()  # obs will be an OrderedDict (default, but you can customize it)
        
        # you can do a "do nothing" action
        act = np.zeros(gym_env.action_space.shape)
        # see danger about curtailment !
        start_, end_ = gym_env.action_space.get_indexes("curtail")
        ## real version, not in the space... (write an issue if it's a problem for you)
        act[start_:end_] = -1
        ## version "in the space"
        # act[start_:end_] = 1
        print(gym_env.action_space.from_gym(act))
        obs, reward, done, truncated, info = gym_env.step(act)
        
        # you can also do a random action:
        act = gym_env.action_space.sample()
        print(gym_env.action_space.from_gym(act))
        obs, reward, done, truncated, info = gym_env.step(act)
        
        # you can do an action on say redispatch (for example)
        act = np.zeros(gym_env.action_space.shape)
        key = "redispatch"  # "redispatch", "curtail", "set_storage" (but there is no storage on this env)
        start_, end_ = gym_env.action_space.get_indexes(key)
        act[start_:end_] = np.random.uniform(high=1, low=-1, size=env.gen_redispatchable.sum())  # the dispatch vector
        print(gym_env.action_space.from_gym(act))
        obs, reward, done, truncated, info = gym_env.step(act)
        
        
    Notes
    -------
    For more customization, this code is roughly equivalent to something like:

    .. code-block:: python

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv
        # this of course will not work... Replace "AGymSpace" with a normal gym space, like Dict, Box, MultiDiscrete etc.
        from gym.spaces import AGymSpace
        gym_env = GymEnv(env)

        class MyCustomActionSpace(AGymSpace):
            def __init__(self, whatever, you, want):
                # do as you please here
                pass
                # don't forget to initialize the base class
                AGymSpace.__init__(self, see, gym, doc, as, to, how, to, initialize, it)
                # eg. MultiDiscrete.__init__(self, nvec=...)

            def from_gym(self, gym_action):
                # this is this very same function that you need to implement
                # it should have this exact name, take only one action (member of your gym space) as input
                # and return a grid2op action
                return TheGymAction_ConvertedTo_Grid2op_Action
                # eg. return np.concatenate((obs.gen_p * 0.1, np.sqrt(obs.load_p))

        gym_env.action_space.close()
        gym_env.action_space = MyCustomActionSpace(whatever, you, wanted)

    And you can implement pretty much anything in the "from_gym" function.

    """

    def __init__(
        self,
        grid2op_action_space,
        attr_to_keep=ALL_ATTR_CONT,
        add=None,
        multiply=None,
        functs=None,
    ):
        if not isinstance(grid2op_action_space, ActionSpace):
            raise RuntimeError(
                f"Impossible to create a BoxGymActSpace without providing a "
                f"grid2op action_space. You provided {type(grid2op_action_space)}"
                f'as the "grid2op_action_space" attribute.'
            )
        check_gym_version(type(self)._gymnasium)
        if attr_to_keep == ALL_ATTR_CONT:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {
                el for el in attr_to_keep if grid2op_action_space.supports_type(el)
            }

        for el in ATTR_DISCRETE:
            if el in attr_to_keep:
                warnings.warn(
                    f'The class "BoxGymActSpace" should mainly be used to consider only continuous '
                    f"actions (eg. redispatch, set_storage or curtail). Though it is possible to use "
                    f'"{el}" when building it, be aware that this discrete action will be treated '
                    f'as continuous. Consider using the "MultiDiscreteActSpace" for these attributes.'
                )
        
        if not attr_to_keep:
            raise Grid2OpException("This would be an empty action set. ")
        
        self._attr_to_keep = sorted(attr_to_keep)

        act_sp_cls = type(grid2op_action_space)
        self._act_space = copy.deepcopy(grid2op_action_space)

        low_gen = -1.0 * act_sp_cls.gen_max_ramp_down[act_sp_cls.gen_redispatchable]
        high_gen = 1.0 * act_sp_cls.gen_max_ramp_up[act_sp_cls.gen_redispatchable]
        nb_redisp = act_sp_cls.gen_redispatchable.sum()
        nb_curtail = act_sp_cls.gen_renewable.sum()
        curtail = np.full(shape=(nb_curtail,), fill_value=0.0, dtype=dt_float)
        curtail_mw = np.full(shape=(nb_curtail,), fill_value=0.0, dtype=dt_float)
        self._dict_properties = {
            "set_line_status": (
                np.full(shape=(act_sp_cls.n_line,), fill_value=-1, dtype=dt_int),
                np.full(shape=(act_sp_cls.n_line,), fill_value=1, dtype=dt_int),
                (act_sp_cls.n_line,),
                dt_int,
            ),
            "change_line_status": (
                np.full(shape=(act_sp_cls.n_line,), fill_value=0, dtype=dt_int),
                np.full(shape=(act_sp_cls.n_line,), fill_value=1, dtype=dt_int),
                (act_sp_cls.n_line,),
                dt_int,
            ),
            "set_bus": (
                np.full(shape=(act_sp_cls.dim_topo,), fill_value=-1, dtype=dt_int),
                np.full(shape=(act_sp_cls.dim_topo,), fill_value=act_sp_cls.n_busbar_per_sub, dtype=dt_int),
                (act_sp_cls.dim_topo,),
                dt_int,
            ),
            "change_bus": (
                np.full(shape=(act_sp_cls.dim_topo,), fill_value=0, dtype=dt_int),
                np.full(shape=(act_sp_cls.dim_topo,), fill_value=1, dtype=dt_int),
                (act_sp_cls.dim_topo,),
                dt_int,
            ),
            "redispatch": (low_gen, high_gen, (nb_redisp,), dt_float),
            "set_storage": (
                -1.0 * act_sp_cls.storage_max_p_prod,
                1.0 * act_sp_cls.storage_max_p_absorb,
                (act_sp_cls.n_storage,),
                dt_float,
            ),
            "curtail": (
                curtail,
                np.full(shape=(nb_curtail,), fill_value=1.0, dtype=dt_float),
                (nb_curtail,),
                dt_float,
            ),
            "curtail_mw": (
                curtail_mw,
                1.0 * act_sp_cls.gen_pmax[act_sp_cls.gen_renewable],
                (nb_curtail,),
                dt_float,
            ),
            "raise_alarm": (
                np.full(shape=(act_sp_cls.dim_alarms,), fill_value=0, dtype=dt_int),
                np.full(shape=(act_sp_cls.dim_alarms,), fill_value=1, dtype=dt_int),
                (act_sp_cls.dim_alarms,),
                dt_int,
            ),
            "raise_alert": (
                np.full(shape=(act_sp_cls.dim_alerts,), fill_value=0, dtype=dt_int),
                np.full(shape=(act_sp_cls.dim_alerts,), fill_value=1, dtype=dt_int),
                (act_sp_cls.dim_alerts,),
                dt_int,
            ),
        }
        self._key_dict_to_proptype = {
            "set_line_status": dt_int,
            "change_line_status": dt_bool,
            "set_bus": dt_int,
            "change_bus": dt_bool,
            "redispatch": dt_float,
            "set_storage": dt_float,
            "curtail": dt_float,
            "curtail_mw": dt_float,
            "raise_alarm": dt_bool,
            "raise_alert": dt_bool
        }

        if add is not None:
            self._add = {k: np.array(v) for k, v in add.items()}
        else:
            self._add = {}
        if multiply is not None:
            self._multiply = {k: np.array(v) for k, v in multiply.items()}
        else:
            self._multiply = {}
            
        # handle the "functional" part
        self.__func = {}

        self._dims = None
        self._dtypes = None
        if functs is None:
            functs = {}
        low, high, shape, dtype = self._get_info(functs)

        # initialize the base container
        type(self)._BoxType.__init__(self, low=low, high=high, shape=shape, dtype=dtype)
        
        # convert data in `_add` and `_multiply` to the right type
        # self._add = {k: v.astype(self.dtype) for k, v in self._add.items()}
        # self._multiply = {k: v.astype(self.dtype) for k, v in self._multiply.items()}
        self._fix_value_sub_div(self._add, functs)
        self._fix_value_sub_div(self._multiply, functs)
        
    def _get_shape(self, el, functs):
        if el in functs:
            callable_, low_, high_, shape_, dtype_ = functs[el]
        elif el in self._dict_properties:
            # el is an attribute of an observation, for example "load_q" or "topo_vect"
            low_, high_, shape_, dtype_ = self._dict_properties[el]
        return shape_
    
    def _fix_value_sub_div(self, dict_, functs):
        """dict_ is either self._add or self._multiply"""
        keys = list(dict_.keys())
        for k in keys:
            v = dict_[k]
            if isinstance(v, (list, tuple)):
                v = np.array(v).astype(self.dtype)
            else:
                shape = self._get_shape(k, functs)
                v = np.full(shape, fill_value=v, dtype=self.dtype)
            dict_[k] = v
                       
    def _get_info(self, functs):
        low = None
        high = None
        shape = None
        dtype = None
        self._dims = []
        self._dtypes = []
        for el in self._attr_to_keep:
            if el in functs:
                # the attribute name "el" has been put in the functs
                callable_, low_, high_, shape_, dtype_ = functs[el]

                if dtype_ is None:
                    dtype_ = dt_float
                if shape_ is None:
                    raise RuntimeError(
                        f'Error: if you use the "functs" keyword for the action space, '
                        f"you need to provide the shape of the vector you expect. See some "
                        f"examples in the official documentation."
                    )
                if low_ is None:
                    low_ = np.full(shape_, fill_value=-np.inf, dtype=dtype_)
                elif isinstance(low_, float):
                    low_ = np.full(shape_, fill_value=low_, dtype=dtype_)

                if high_ is None:
                    high_ = np.full(shape_, fill_value=np.inf, dtype=dtype_)
                elif isinstance(high_, float):
                    high_ = np.full(shape_, fill_value=high_, dtype=dtype_)

                # simulate a vector in the range low_, high_ and the right shape to test the function given
                # by the user
                vect_right_properties = np.random.uniform(size=shape_)
                finite_both = np.isfinite(low_) & np.isfinite(high_)
                fintte_low = np.isfinite(low_) & ~np.isfinite(high_)
                fintte_high = ~np.isfinite(low_) & np.isfinite(high_)
                vect_right_properties[finite_both] = (
                    vect_right_properties[finite_both]
                    * (high_[finite_both] - low_[finite_both])
                    + low_[finite_both]
                )
                vect_right_properties[fintte_low] += low_[fintte_low]

                try:
                    tmp = callable_(vect_right_properties)
                except Exception as exc_:
                    raise RuntimeError(
                        f'Error for the function your provided with key "{el}". '
                        f"The error was :\n {exc_}"
                    )
                if not isinstance(tmp, BaseAction):
                    raise RuntimeError(
                        f'The function you provided in the "functs" argument for key "{el}" '
                        f"should take a"
                    )

                self.__func[el] = callable_

            elif el in self._dict_properties:
                # el is an attribute of an observation, for example "load_q" or "topo_vect"
                low_, high_, shape_, dtype_ = self._dict_properties[el]
            else:
                li_keys = "\n\t- ".join(
                    sorted(list(self._dict_properties.keys()) + list(self.__func.keys()))
                )
                raise RuntimeError(
                    f'Unknown action attributes "{el}". Supported attributes are: '
                    f"\n\t- {li_keys}"
                )

            # handle the data type
            if dtype is None:
                dtype = dtype_
            else:
                if dtype_ == dt_float:
                    # promote whatever to float anyway
                    dtype = dt_float
                elif dtype_ == dt_int and dtype == dt_bool:
                    # promote bool to int
                    dtype = dt_int

            # handle the shape
            if shape is None:
                shape = shape_
            else:
                shape = (shape[0] + shape_[0],)

            # handle low / high
            # NB: the formula is: glop = gym * multiply + add                
            if el in self._add:
                low_ =  1.0 * low_.astype(dtype)
                high_ =  1.0 * high_.astype(dtype)
                low_ -= self._add[el]
                high_ -= self._add[el]
                
            if el in self._multiply:
                # special case if a 0 were entered
                arr_ = 1.0 * self._multiply[el]
                is_nzero = np.abs(arr_) >= 1e-7
                
                low_ =  1.0 * low_.astype(dtype)
                high_ =  1.0 * high_.astype(dtype)
                low_[is_nzero] /= arr_[is_nzero]
                high_[is_nzero] /= arr_[is_nzero]

            # "fix" the low / high : they can be inverted if self._multiply < 0. for example
            tmp_l = copy.deepcopy(low_)
            tmp_h = copy.deepcopy(high_)
            low_ = np.minimum(tmp_h, tmp_l)
            high_ = np.maximum(tmp_h, tmp_l)

            if low is None:
                low = low_
                high = high_
            else:
                low = np.concatenate((low.astype(dtype), low_.astype(dtype))).astype(
                    dtype
                )
                high = np.concatenate((high.astype(dtype), high_.astype(dtype))).astype(
                    dtype
                )

            # remember where this need to be stored
            self._dims.append(shape[0])
            self._dtypes.append(dtype_)

        return low, high, shape, dtype

    def _handle_attribute(self, res, gym_act_this, attr_nm):
        """
        INTERNAL

        TODO

        Parameters
        ----------
        res
        gym_act_this
        attr_nm

        Returns
        -------

        """
        if attr_nm in self._multiply:
            gym_act_this *= self._multiply[attr_nm]
        if attr_nm in self._add:
            gym_act_this += self._add[attr_nm]

        if attr_nm == "curtail":
            gym_act_this_ = np.full(
                self._act_space.n_gen, fill_value=np.NaN, dtype=dt_float
            )
            gym_act_this_[self._act_space.gen_renewable] = gym_act_this
            gym_act_this = gym_act_this_
        elif attr_nm == "curtail_mw":
            gym_act_this_ = np.full(
                self._act_space.n_gen, fill_value=np.NaN, dtype=dt_float
            )
            gym_act_this_[self._act_space.gen_renewable] = gym_act_this
            gym_act_this = gym_act_this_
        elif attr_nm == "redispatch":
            gym_act_this_ = np.zeros(self._act_space.n_gen, dtype=dt_float)
            gym_act_this_[self._act_space.gen_redispatchable] = gym_act_this
            gym_act_this = gym_act_this_

        setattr(res, attr_nm, gym_act_this)
        return res

    def get_indexes(self, key: str) -> Tuple[int, int]:
        """Allows to retrieve the indexes of the gym action that
        are concerned by the attribute name `key` given in input.

        Parameters
        ----------
        key : str
            the attribute name (*eg* "set_storage" or "redispatch")

        Returns
        -------
        Tuple[int, int]
            _description_

        Examples
        --------
        
        You can use it like:
        
        .. code-block:: python
        
            gym_env = ... # an environment with a BoxActSpace
            
            act = np.zeros(gym_env.action_space.shape)
            key = "redispatch"  # "redispatch", "curtail", "set_storage"
            start_, end_ = gym_env.action_space.get_indexes(key)
            act[start_:end_] = np.random.uniform(high=1, low=-1, size=env.gen_redispatchable.sum())
            # act only modifies the redispatch with the input given (here a uniform redispatching between -1 and 1)
            
        """
        error_msg =(f"Impossible to use the grid2op action property \"{key}\""
                    f"with this action space.")
        if key not in self._attr_to_keep:
            raise Grid2OpException(error_msg)
        prev = 0
        for attr_nm, where_to_put in zip(
            self._attr_to_keep, self._dims
        ):
            if attr_nm == key:
                return prev, where_to_put
            prev = where_to_put
        raise Grid2OpException(error_msg)
        
    def from_gym(self, gym_act):
        """
        This is the function that is called to transform a gym action (in this case a numpy array!)
        sent by the agent
        and convert it to a grid2op action that will be sent to the underlying grid2op environment.

        Parameters
        ----------
        gym_act: ``numpy.ndarray``
            the gym action

        Returns
        -------
        grid2op_act: :class:`grid2op.Action.BaseAction`
            The corresponding grid2op action.

        """
        res = self._act_space()
        prev = 0
        for attr_nm, where_to_put, dtype in zip(
            self._attr_to_keep, self._dims, self._dtypes
        ):
            this_part = 1 * gym_act[prev:where_to_put]
            if attr_nm in self.__func:
                glop_act_tmp = self.__func[attr_nm](this_part)
                res += glop_act_tmp
            elif hasattr(res, attr_nm):
                glop_dtype = self._key_dict_to_proptype[attr_nm]
                if glop_dtype == dt_int:
                    # convert floating point actions to integer.
                    # NB: i round first otherwise it is cut.
                    this_part = np.round(this_part, 0).astype(dtype)
                elif glop_dtype == dt_bool:
                    # convert floating point actions to bool.
                    # NB: it's important here the numbers are between 0 and 1
                    this_part = (this_part >= 0.5).astype(dt_bool)
                if this_part.shape and this_part.shape[0]:
                    # only update the attribute if there is actually something to update
                    self._handle_attribute(res, this_part, attr_nm)
            else:
                raise RuntimeError(f'Unknown attribute "{attr_nm}".')
            prev = where_to_put
        return res

    def close(self):
        pass

    def normalize_attr(self, attr_nm: str):
        """
        This function normalizes the part of the space
        that corresponds to the attribute `attr_nm`.

        The normalization consists in having a vector between 0. and 1.
        It is achieved by:
        
        - dividing by the range (high - low)
        - adding the minimum value (low).

        .. note::
            It only affects continuous attribute. No error / warnings are
            raised if you attempt to use it on a discrete attribute.

        .. warning::
            This normalization relies on the `high` and `low` attribute. It cannot be done if
            the attribute is not bounded (for example when its maximum limit is `np.inf`). A warning
            is raised in this case.
            
        Parameters
        ----------
        attr_nm : str
            The name of the attribute to normalize
            
        """
        if attr_nm in self._multiply or attr_nm in self._add:
            raise Grid2OpException(
                f"Cannot normalize attribute \"{attr_nm}\" that you already "
                "modified with either `add` or `multiply` (action space)."
            )
        prev = 0
        for attr_tmp, where_to_put, dtype in zip(
            self._attr_to_keep, self._dims, self._dtypes
        ):
            if attr_tmp == attr_nm and dtype == dt_float:
                curr_high = 1.0 * self.high[prev:where_to_put]
                curr_low = 1.0 * self.low[prev:where_to_put]
                finite_high = np.isfinite(curr_high)
                finite_low = np.isfinite(curr_high)
                both_finite = finite_high & finite_low
                both_finite &= curr_high > curr_low

                if (~both_finite).any():
                    warnings.warn(f"The normalization of attribute \"{attr_tmp}\" cannot be performed entirely as "
                                  f"there are some non finite value, or `high == `low` "
                                  f"for some components.")
                    
                self._multiply[attr_nm] = np.ones(curr_high.shape, dtype=dtype)
                self._add[attr_nm] = np.zeros(curr_high.shape, dtype=dtype)

                self._multiply[attr_nm][both_finite] = (
                    curr_high[both_finite] - curr_low[both_finite]
                )
                self._add[attr_nm][both_finite] += curr_low[both_finite]
                self.high[prev:where_to_put][both_finite] = 1.0
                self.low[prev:where_to_put][both_finite] = 0.0
                break
            prev = where_to_put


if GYM_AVAILABLE:
    from gym.spaces import Box as LegGymBox
    from grid2op.gym_compat.base_gym_attr_converter import BaseLegacyGymAttrConverter
    BoxLegacyGymActSpace = type("BoxLegacyGymActSpace",
                                (__AuxBoxGymActSpace, LegGymBox, ),
                                {"_gymnasium": False,
                                 "_BaseGymAttrConverterType": BaseLegacyGymAttrConverter,
                                 "_BoxType": LegGymBox,
                                 "__module__": __name__})
    BoxLegacyGymActSpace.__doc__ = __AuxBoxGymActSpace.__doc__
    BoxGymActSpace = BoxLegacyGymActSpace
    BoxGymActSpace.__doc__ = __AuxBoxGymActSpace.__doc__
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Box
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    BoxGymnasiumActSpace = type("BoxGymnasiumActSpace",
                                (__AuxBoxGymActSpace, Box, ),
                                {"_gymnasium": True,
                                 "_BaseGymAttrConverterType": BaseGymnasiumAttrConverter,
                                 "_BoxType": Box,
                                 "__module__": __name__})
    BoxGymnasiumActSpace.__doc__ = __AuxBoxGymActSpace.__doc__
    BoxGymActSpace = BoxGymnasiumActSpace
    BoxGymActSpace.__doc__ = __AuxBoxGymActSpace.__doc__
