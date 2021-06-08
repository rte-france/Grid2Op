# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
from gym.spaces import Discrete


from grid2op.Action import ActionSpace
from grid2op.Converter import IdToAct

from grid2op.gym_compat.utils import ALL_ATTR, ATTR_DISCRETE
# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part


class DiscreteActSpace(Discrete):
    """
    TODO the documentation of this class is in progress.

    This class allows to convert a grid2op action space into a gym "Discrete". This means that the action are
    labeled, and instead of describing the action itself, you provide only its ID.

    It is related to the :class:`MultiDiscreteActSpace` but compared to this other representation, it
    does not allow to do "multiple actions". Typically, if you use the snippets below:

    .. code-block:: python

        import grid2op
        env_name = ...
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv, MultiDiscreteActSpace, DiscreteActSpace
        gym_env1 = GymEnv(env)
        gym_env2 = GymEnv(env)

        gym_env1.action_space = MultiDiscreteActSpace(env.action_space,
                                                      attr_to_keep=['redispatch', "curtail", "one_sub_set"])
        gym_env2.action_space = MultiDiscreteActSpace(env.action_space,
                                                      attr_to_keep=['redispatch', "curtail", "set_bus"])


    Then at each step, `gym_env1` will allow to perform a redispatching action (on any number of generators),
    a curtailment
    action (on any number of generators) __**AND**__  changing the topology at one substation. But at each
    steps, the agent should predicts lots of "number".

    On the other hand, at each step, the agent for `gym_env2` will have to predict a single integer (which is
    usually the case in most RL environment) but it this action will affect redispatching on a single generator,
    perform curtailment on a single generator __**OR**__  changing the topology at one substation. But at each
    steps, the agent should predicts lots of "number".

    The action set is then largely  constrained compared to the :class:`MultiDiscreteActSpace`

    .. note::
        This class is really closely related to the :class:`grid2op.Converter.IdToAct`. It basically "maps"
        this "IdToAct" into a type of gym space, which, in this case, will be a Discrete one.

    Examples
    --------

    We recommend to use it like:

    .. code-block:: python

        import grid2op
        env_name = ...
        env = grid2op.make(env_name)

        from grid2op.gym_compat import GymEnv, MultiDiscreteActSpace, DiscreteActSpace
        gym_env = GymEnv(env)

        gym_env.observation_space = DiscreteActSpace(env.observation_space,
                                                     attr_to_keep=['redispatch', "curtail", "set_bus"])

    The possible attribute you can provide in the "attr_to_keep" are:

    - "set_line_status"
    - "change_line_status"
    - "set_bus": corresponds to changing the topology using the "set_bus" (equivalent to the
      "one_sub_set" keyword in the "attr_to_keep" of the :class:`MultiDiscreteActSpace`)
    - "change_bus": corresponds to changing the topology using the "change_bus" (equivalent to the
      "one_sub_change" keyword in the "attr_to_keep" of the :class:`MultiDiscreteActSpace`)
    - "redispatch"
    - "set_storage"
    - "curtail"
    - "curtail_mw" (same effect as "curtail")

    """
    def __init__(self,
                 grid2op_action_space,
                 attr_to_keep=ALL_ATTR,
                 nb_bins=None,
                 ):

        if not isinstance(grid2op_action_space, ActionSpace):
            raise RuntimeError(f"Impossible to create a BoxGymActSpace without providing a "
                               f"grid2op action_space. You provided {type(grid2op_action_space)}"
                               f"as the \"grid2op_action_space\" attribute.")

        if nb_bins is None:
            nb_bins = {"redispatch": 7, "set_storage": 7, "curtail": 7}

        act_sp = grid2op_action_space
        self.action_space = copy.deepcopy(act_sp)

        if attr_to_keep == ALL_ATTR:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {el for el in attr_to_keep if grid2op_action_space.supports_type(el)}

        for el in attr_to_keep:
            if el not in ATTR_DISCRETE:
                warnings.warn(f"The class \"DiscreteActSpace\" should mainly be used to consider only discrete "
                              f"actions (eg. set_line_status, set_bus or change_bus). Though it is possible to use "
                              f"\"{el}\" when building it, be aware that this continuous action will be treated "
                              f"as discrete by splitting it into bins. "
                              f"Consider using the \"BoxGymActSpace\" for these attributes."
                              )

        self._attr_to_keep = sorted(attr_to_keep)
        self._nb_bins = nb_bins

        self.dict_properties = {
            "set_line_status": act_sp.get_all_unitary_line_set,
            "change_line_status": act_sp.get_all_unitary_line_change,
            "set_bus": act_sp.get_all_unitary_topologies_set,
            "change_bus": act_sp.get_all_unitary_topologies_change,
            "redispatch": act_sp.get_all_unitary_redispatch,
            "set_storage": act_sp.get_all_unitary_storage,
            "curtail": act_sp.get_all_unitary_curtail,
            "curtail_mw": act_sp.get_all_unitary_curtail,
            "raise_alarm": act_sp.get_all_unitary_alarm,
        }

        self.converter = None
        n_act = self._get_info()
        
        # initialize the base container
        Discrete.__init__(self, n=n_act)

    def _get_info(self):
        converter = IdToAct(self.action_space)
        li_act = [self.action_space()]
        for attr_nm in self._attr_to_keep:
            if attr_nm in self.dict_properties:
                if attr_nm not in self._nb_bins:
                    li_act += self.dict_properties[attr_nm](self.action_space)
                else:
                    if attr_nm == "curtail" or attr_nm == "curtail_mw":
                        li_act += self.dict_properties[attr_nm](self.action_space, num_bin=self._nb_bins[attr_nm])
                    else:
                        li_act += self.dict_properties[attr_nm](self.action_space,
                                                                num_down=self._nb_bins[attr_nm],
                                                                num_up=self._nb_bins[attr_nm])
            else:
                li_keys = '\n\t- '.join(sorted(list(self.dict_properties.keys())))
                raise RuntimeError(f"Unknown action attributes \"{attr_nm}\". Supported attributes are: "
                                   f"\n\t- {li_keys}")

        converter.init_converter(li_act)
        self.converter = converter
        return self.converter.n

    def from_gym(self, gym_act):
        """
        This is the function that is called to transform a gym action (in this case a numpy array!)
        sent by the agent
        and convert it to a grid2op action that will be sent to the underlying grid2op environment.

        Parameters
        ----------
        gym_act: ``int````
            the gym action (a single integer for this action space)

        Returns
        -------
        grid2op_act: :class:`grid2op.Action.BaseAction`
            The corresponding grid2op action.

        """
        res = self.converter.all_actions[int(gym_act)]
        return res
