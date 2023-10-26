# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
import numpy as np

from grid2op.Action import ActionSpace
from grid2op.dtypes import dt_int, dt_bool, dt_float

from grid2op.gym_compat.utils import (ALL_ATTR,
                                      ATTR_DISCRETE,
                                      check_gym_version,
                                      GYM_AVAILABLE,
                                      GYMNASIUM_AVAILABLE)


class __AuxMultiDiscreteActSpace:
    """
    This class allows to convert a grid2op action space into a gym "MultiDiscrete". This means that the action are
    labeled, and instead of describing the action itself, you provide only its ID.

    .. note::
        This action space is particularly suited for represented discrete actions.

        It is possible to represent continuous actions with it. In that case, the continuous actions are "binarized"
        thanks to the :class:`ContinuousToDiscreteConverter`. Feel free to consult its documentation for
        more information.

    In this case it will extract all the features in all the action with:

    - "set_line_status": `n_line` dimensions, each containing 3 choices "DISCONNECT", "DONT AFFECT", "FORCE CONNECTION"
      and affecting the powerline status (connected / disconnected)
    - "change_line_status":  `n_line` dimensions, each containing 2 elements "CHANGE", "DONT CHANGE" and
      affecting the powerline status (connected / disconnected)
    - "set_bus": `dim_topo` dimensions, each containing 4 choices: "DISCONNECT", "DONT AFFECT", "CONNECT TO BUSBAR 1",
      or "CONNECT TO BUSBAR 2" and affecting to which busbar an object is connected
    - "change_bus": `dim_topo` dimensions, each containing 2 choices: "CHANGE", "DONT CHANGE" and affect
      to which busbar an element is connected
    - "redispatch": `sum(env.gen_redispatchable)` dimensions, each containing a certain number of choices depending on the value
      of the keyword argument `nb_bins["redispatch"]` (by default 7).
    - "curtail": `sum(env.gen_renewable)` dimensions, each containing a certain number of choices depending on the value
      of the keyword argument `nb_bins["curtail"]` (by default 7). This is
      the "conversion to discrete action"
      of the curtailment action.
    - "curtail_mw": `sum(env.gen_renewable)` dimensions, completely equivalent to "curtail" for this representation. 
      This is the "conversion to discrete action" of the curtailment action.
    - "set_storage": `n_storage` dimensions, each containing a certain number of choices depending on the value
      of the keyword argument `nb_bins["set_storage"]` (by default 7). This is the "conversion to discrete action"
      of the action on storage units.
    - "raise_alarm": TODO
    - "raise_alert": TODO


    We offer some extra customization, with the keywords:

    - "sub_set_bus": `n_sub` dimension. This type of representation encodes each different possible combination
      of elements that are possible at each substation. The choice at each component depends on the element connected
      at this substation. Only configurations that will not lead to straight game over will be generated.
    - "sub_change_bus": `n_sub` dimension. Same comment as for "sub_set_bus"
    - "one_sub_set": 1 single dimension. This type of representation differs from the previous one only by the fact
      that each step you can perform only one single action on a single substation (so unlikely to be illegal).
    - "one_sub_change": 1 single dimension. Same as above.

    .. warning::

        We recommend to use either "set" or "change" way to look at things (**ie** either you want to target
        a given state -in that case use "sub_set_bus", "line_set_status", "one_sub_set", or "set_bus" __**OR**__ you
        prefer
        reasoning in terms of "i want to change this or that" in that case use "sub_change_bus",
        "line_change_status", "one_sub_change" or "change_bus".

        Combining a "set" and "change" on the same element will most likely lead to an "ambiguous action". Indeed
        what grid2op can do if you "tell element A to go to bus 1" and "tell the same element A to switch to bus 2 if it was
        to 1 and to move to bus 1 if it was on bus 2". It's not clear at all (hence the "ambiguous").

        No error will be thrown if you mix this, this is your absolute right, be aware it might not
        lead to the result you expect though.

    .. note::

        The arguments "set_bus", "sub_set_bus" and "one_sub_set" will all perform "set_bus" actions. The only
        difference if "how you represent these actions":

        - In "set_bus" each component represent a single element of the grid. When you sample an action
          with this keyword you will possibly change all the elements of the grid at once (this is likely to
          be illega). Nothing prevents you to perform "weird" stuff, for example disconnecting a load or a generator
          (which is straight game over) or having a load or a generator that will be "alone" on a busbar (which
          will also lead to a straight game over). You can do anything with it, but as always "A great power
          comes with a great responsibility".
        - In "sub_set_bus" each component represent a substation of the grid. When you sample an action
          from this, you will possibly change all the elements of the grid at once (because you can act
          on all the substation at the same time). As opposed to "set_bus" however this constraint the action
          space to "action that will not lead directly to a game over", in practice.
        - In "one_sub_set": the single component represent the whole grid. When you sample an action
          with this, you will sample a single action acting on a single substation. You will not be able to act
          on multiple substation with this.

        For this reason, we also do not recommend using only one of these arguments and only provide
        only one of "set_bus", "sub_set_bus" and "one_sub_set". Again, no error will be thrown if you mix them
        but be warned that the resulting behaviour might not be what you expect.

    .. warning::

        The same as above holds for "change_bus", "sub_change_bus" and "one_sub_change": Use only one of these !

    .. danger::
        The keys `set_bus` and `change_bus` does not have the same meaning between this representation of the
        action and the DiscreteActSpace.
    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`MultiDiscreteActSpace` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`MultiDiscreteActSpaceGymnasium`), otherwise it will
          inherit from gym (and will be exactly :class:`MultiDiscreteActSpaceLegacyGym`)
        - :class:`MultiDiscreteActSpaceGymnasium` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`MultiDiscreteActSpaceLegacyGym` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    Examples
    --------
    If you simply want to use it you can do:

    .. code-block:: python

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv, MultiDiscreteActSpace
        gym_env = GymEnv(env)

        gym_env.action_space = MultiDiscreteActSpace(env.action_space)

    You can select the attribute you want to keep, for example:

    .. code-block:: python

        gym_env.action_space = MultiDiscreteActSpace(env.observation_space,
                                                     attr_to_keep=['redispatch', "curtail", "sub_set_bus"])

    You can also apply some basic transformation when you "discretize" continuous action

    .. code-block:: python

        gym_env.action_space = MultiDiscreteActSpace(env.observation_space,
                                                     attr_to_keep=['redispatch', "curtail", "sub_set_bus"],
                                                     nb_bins={"redispatch": 3, "curtail": 17},
                                                     )

    By default it is "discretized" in 7 different "bins". The more "bins" there will be, the more "precise"
    you can be in your control, but the higher the dimension of the action space.

    """

    ATTR_CHANGE = 0
    ATTR_SET = 1
    ATTR_NEEDBUILD = 2
    ATTR_NEEDBINARIZED = 3

    def __init__(self, grid2op_action_space, attr_to_keep=ALL_ATTR, nb_bins=None):
        check_gym_version(type(self)._gymnasium)
        if not isinstance(grid2op_action_space, ActionSpace):
            raise RuntimeError(
                f"Impossible to create a BoxGymActSpace without providing a "
                f"grid2op action_space. You provided {type(grid2op_action_space)}"
                f'as the "grid2op_action_space" attribute.'
            )

        if nb_bins is None:
            nb_bins = {"redispatch": 7, "set_storage": 7, "curtail": 7, "curtail_mw": 7}

        if attr_to_keep == ALL_ATTR:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {
                el for el in attr_to_keep if grid2op_action_space.supports_type(el)
            }

        for el in attr_to_keep:
            if el not in ATTR_DISCRETE:
                warnings.warn(
                    f'The class "MultiDiscreteActSpace" should mainly be used to consider only discrete '
                    f"actions (eg. set_line_status, set_bus or change_bus). Though it is possible to use "
                    f'"{el}" when building it, be aware that this continuous action will be treated '
                    f"as discrete by splitting it into bins. "
                    f'Consider using the "BoxGymActSpace" for these attributes.'
                )

        self._attr_to_keep = sorted(attr_to_keep)

        act_sp = grid2op_action_space
        self._act_space = copy.deepcopy(grid2op_action_space)

        low_gen = -1.0 * act_sp.gen_max_ramp_down
        high_gen = 1.0 * act_sp.gen_max_ramp_up
        low_gen[~act_sp.gen_redispatchable] = 0.0
        high_gen[~act_sp.gen_redispatchable] = 0.0

        # nb, dim, []
        self.dict_properties = {
            "set_line_status": (
                [3 for _ in range(act_sp.n_line)],
                act_sp.n_line,
                self.ATTR_SET,
            ),
            "change_line_status": (
                [2 for _ in range(act_sp.n_line)],
                act_sp.n_line,
                self.ATTR_CHANGE,
            ),
            "set_bus": (
                [4 for _ in range(act_sp.dim_topo)],
                act_sp.dim_topo,
                self.ATTR_SET,
            ),
            "change_bus": (
                [2 for _ in range(act_sp.dim_topo)],
                act_sp.dim_topo,
                self.ATTR_CHANGE,
            ),
            "raise_alarm": (
                [2 for _ in range(act_sp.dim_alarms)],
                act_sp.dim_alarms,
                self.ATTR_CHANGE,
            ),
            "raise_alert": (
                [2 for _ in range(act_sp.dim_alerts)],
                act_sp.dim_alerts,
                self.ATTR_CHANGE,
            ),
            "sub_set_bus": (
                None,
                act_sp.n_sub,
                self.ATTR_NEEDBUILD,
            ),  # dimension will be computed on the fly, if the stuff is used
            "sub_change_bus": (
                None,
                act_sp.n_sub,
                self.ATTR_NEEDBUILD,
            ),  # dimension will be computed on the fly, if the stuff is used
            "one_sub_set": (
                None,
                1,
                self.ATTR_NEEDBUILD,
            ),  # dimension will be computed on the fly, if the stuff is used
            "one_sub_change": (
                None,
                1,
                self.ATTR_NEEDBUILD,
            ),  # dimension will be computed on the fly, if the stuff is used
        }
        self._nb_bins = nb_bins
        for el in ["redispatch", "set_storage", "curtail", "curtail_mw"]:
            if el in attr_to_keep:
                if el not in nb_bins:
                    raise RuntimeError(
                        f'The attribute you want to keep "{el}" is not present in the '
                        f'"nb_bins". This attribute is continuous, you have to specify in how '
                        f"how to convert it to a discrete space. See the documentation "
                        f"for more information."
                    )
                nb_redispatch = act_sp.gen_redispatchable.sum()
                nb_renew = act_sp.gen_renewable.sum()
                if el == "redispatch":
                    self.dict_properties[el] = (
                        [nb_bins[el] for _ in range(nb_redispatch)],
                        nb_redispatch,
                        self.ATTR_NEEDBINARIZED,
                    )
                elif el == "curtail" or el == "curtail_mw":
                    self.dict_properties[el] = (
                        [nb_bins[el] for _ in range(nb_renew)],
                        nb_renew,
                        self.ATTR_NEEDBINARIZED,
                    )
                elif el == "set_storage":
                    self.dict_properties[el] = (
                        [nb_bins[el] for _ in range(act_sp.n_storage)],
                        act_sp.n_storage,
                        self.ATTR_NEEDBINARIZED,
                    )
                else:
                    raise RuntimeError(f'Unknown attribute "{el}"')

        self._dims = None
        self._functs = None  # final functions that is applied to the gym action to map it to a grid2Op action
        self._binarizers = None  # contains all the stuff to binarize the data
        self._types = None
        nvec = self._get_info()

        # initialize the base container
        type(self)._MultiDiscreteType.__init__(self, nvec=nvec)

    @staticmethod
    def _funct_set(vect):
        # gym encodes:
        # for set_bus: 0 -> -1, 1-> 0 (don't change)), 2-> 1, 3 -> 2
        # for set_status: 0 -> -1, 1-> 0 (don't change)), 2-> 1 [3 do not exist for set_line_status !]
        vect -= 1
        return vect

    @staticmethod
    def _funct_change(vect):
        # gym encodes 0 -> False, 1 -> True
        vect = vect.astype(dt_bool)
        return vect

    def _funct_substations(self, orig_act, attr_nm, vect):
        """
        Used for "sub_set_bus" and "sub_change_bus"
        """
        vect_act = self._sub_modifiers[attr_nm]
        for sub_id, act_id in enumerate(vect):
            orig_act += vect_act[sub_id][act_id]

    def _funct_one_substation(self, orig_act, attr_nm, vect):
        """
        Used for "one_sub_set" and "one_sub_change"
        """
        orig_act += self._sub_modifiers[attr_nm][int(vect)]

    def _get_info(self):
        nvec = None
        self._dims = []
        self._functs = []
        self._binarizers = {}
        self._sub_modifiers = {}
        self._types = []
        box_space = None
        dim = 0
        for el in self._attr_to_keep:
            if el in self.dict_properties:
                nvec_, dim_, type_ = self.dict_properties[el]
                if type_ == self.ATTR_CHANGE:
                    # I can convert them directly into discrete attributes because it's a
                    # recognize "change" attribute
                    funct = self._funct_change
                elif type_ == self.ATTR_SET:
                    # I can convert them directly into discrete attributes because it's a
                    # recognize "set" attribute
                    funct = self._funct_set
                elif type_ == self.ATTR_NEEDBINARIZED:
                    # base action was continuous, i need to convert it to discrete action thanks
                    # to "binarization", that is done automatically here
                    # from grid2op.gym_compat.box_gym_actspace import BoxGymActSpace
                    # from grid2op.gym_compat.continuous_to_discrete import (
                        # ContinuousToDiscreteConverter,
                    # )

                    if box_space is None:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            box_space = type(self)._BoxGymActSpaceType(
                                self._act_space,
                                attr_to_keep=[
                                    "redispatch",
                                    "set_storage",
                                    "curtail",
                                    "curtail_mw",
                                ],
                            )

                    if el not in box_space._dict_properties:
                        raise RuntimeError(
                            f"Impossible to dertmine lowest and maximum value for "
                            f'key "{el}".'
                        )
                    low_, high_, shape_, dtype_ = box_space._dict_properties[el]
                    tmp_box = type(self)._BoxType(low=low_, high=high_, dtype=dtype_)
                    tmp_binarizer = type(self)._ContinuousToDiscreteConverterType(
                        init_space=tmp_box, nb_bins=self._nb_bins[el]
                    )
                    self._binarizers[el] = tmp_binarizer
                    funct = tmp_binarizer.gym_to_g2op
                elif type_ == self.ATTR_NEEDBUILD:
                    # attributes comes from substation manipulation, i need to build the entire space
                    nvec_ = []
                    self._sub_modifiers[el] = []
                    if el == "sub_set_bus":
                        # one action per substations, using "set"
                        for sub_id in range(self._act_space.n_sub):
                            act_this_sub = [self._act_space()]
                            act_this_sub += (
                                self._act_space.get_all_unitary_topologies_set(
                                    self._act_space, sub_id=sub_id
                                )
                            )
                            nvec_.append(len(act_this_sub))
                            self._sub_modifiers[el].append(act_this_sub)
                        funct = self._funct_substations
                    elif el == "sub_change_bus":
                        # one action per substation, using "change"
                        for sub_id in range(self._act_space.n_sub):
                            acts_this_sub = [self._act_space()]
                            acts_this_sub += (
                                self._act_space.get_all_unitary_topologies_change(
                                    self._act_space, sub_id=sub_id
                                )
                            )
                            nvec_.append(len(acts_this_sub))
                            self._sub_modifiers[el].append(acts_this_sub)
                        funct = self._funct_substations
                    elif el == "one_sub_set":
                        # an action change only one substation, using "set"
                        self._sub_modifiers[
                            el
                        ] = self._act_space.get_all_unitary_topologies_set(
                            self._act_space
                        )
                        funct = self._funct_one_substation
                        nvec_ = [len(self._sub_modifiers[el])]
                    elif el == "one_sub_change":
                        # an action change only one substation, using "change"
                        self._sub_modifiers[
                            el
                        ] = self._act_space.get_all_unitary_topologies_change(
                            self._act_space
                        )
                        funct = self._funct_one_substation
                        nvec_ = [len(self._sub_modifiers[el])]
                    else:
                        raise RuntimeError(
                            f'Unsupported attribute "{el}" when dealing with '
                            f"action on substation"
                        )

                else:
                    raise RuntimeError(f"Unknown way to build the action.")
            else:
                li_keys = "\n\t- ".join(sorted(list(self.dict_properties.keys())))
                raise RuntimeError(
                    f'Unknown action attributes "{el}". Supported attributes are: '
                    f"\n\t- {li_keys}"
                )
            dim += dim_
            if nvec is not None:
                nvec += nvec_
            else:
                nvec = nvec_
            self._dims.append(dim)
            self._functs.append(funct)
            self._types.append(type_)
        return nvec

    def _handle_attribute(self, res, gym_act_this, attr_nm, funct, type_):
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
        # TODO code that !
        vect = 1 * gym_act_this
        if type_ == self.ATTR_NEEDBUILD:
            funct(res, attr_nm, vect)
        else:
            tmp = funct(vect)
            if attr_nm == "redispatch":
                gym_act_this_ = np.full(
                    self._act_space.n_gen, fill_value=np.NaN, dtype=dt_float
                )
                gym_act_this_[self._act_space.gen_redispatchable] = tmp
                tmp = gym_act_this_
            elif attr_nm == "curtail" or attr_nm == "curtail_mw":
                gym_act_this_ = np.full(
                    self._act_space.n_gen, fill_value=np.NaN, dtype=dt_float
                )
                gym_act_this_[self._act_space.gen_renewable] = tmp
                tmp = gym_act_this_
            setattr(res, attr_nm, tmp)
        return res

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
        for attr_nm, where_to_put, funct, type_ in zip(
            self._attr_to_keep, self._dims, self._functs, self._types
        ):
            if not gym_act.shape or not gym_act.shape[0]:
                continue
            this_part = 1 * gym_act[prev:where_to_put]
            if attr_nm in self.dict_properties:
                self._handle_attribute(res, this_part, attr_nm, funct, type_)
            else:
                raise RuntimeError(f'Unknown attribute "{attr_nm}".')
            prev = where_to_put
        return res

    def close(self):
        pass


if GYM_AVAILABLE:
    from gym.spaces import Box as LegacyGymBox, MultiDiscrete as LegacyGymMultiDiscrete
    from grid2op.gym_compat.box_gym_actspace import BoxLegacyGymActSpace
    from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverterLegacyGym
    MultiDiscreteActSpaceLegacyGym = type("MultiDiscreteActSpaceLegacyGym",
                                          (__AuxMultiDiscreteActSpace, LegacyGymMultiDiscrete, ),
                                          {"_gymnasium": False,
                                           "_BoxType": LegacyGymBox,
                                           "_MultiDiscreteType": LegacyGymMultiDiscrete,
                                           "_BoxGymActSpaceType": BoxLegacyGymActSpace,
                                           "_ContinuousToDiscreteConverterType": ContinuousToDiscreteConverterLegacyGym,
                                           "__module__": __name__})
    MultiDiscreteActSpaceLegacyGym.__doc__ = __AuxMultiDiscreteActSpace.__doc__
    MultiDiscreteActSpace = MultiDiscreteActSpaceLegacyGym
    MultiDiscreteActSpace.__doc__ = __AuxMultiDiscreteActSpace.__doc__
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Box, MultiDiscrete
    from grid2op.gym_compat.box_gym_actspace import BoxGymnasiumActSpace
    from grid2op.gym_compat.continuous_to_discrete import  ContinuousToDiscreteConverterGymnasium
    MultiDiscreteActSpaceGymnasium = type("MultiDiscreteActSpaceGymnasium",
                                          (__AuxMultiDiscreteActSpace, MultiDiscrete, ),
                                          {"_gymnasium": True,
                                           "_BoxType": Box,
                                           "_MultiDiscreteType": MultiDiscrete,
                                           "_BoxGymActSpaceType": BoxGymnasiumActSpace,
                                           "_ContinuousToDiscreteConverterType": ContinuousToDiscreteConverterGymnasium,
                                           "__module__": __name__})
    MultiDiscreteActSpaceGymnasium.__doc__ = __AuxMultiDiscreteActSpace.__doc__
    MultiDiscreteActSpace = MultiDiscreteActSpaceGymnasium
    MultiDiscreteActSpace.__doc__ = __AuxMultiDiscreteActSpace.__doc__
