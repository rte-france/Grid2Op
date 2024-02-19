# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np

from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_int, dt_float

# TODO: use the "last_obs" and the "change_bus" in case of "set_bus" not available


class ConnectivityConverter(Converter):
    """
    In this converter, you have as many output as pairs of object that can be connected, and your model is asked
    to output 0 if he wants these elements disconnected and 1 if he wants them connected.

    This type of modeling is rather hard to "get working" the first time, especially because some "conflict" might
    appear. For example, consider three objects (line for example) on a given substation. You can chose to "connect
    A and B", connect "B and C" but "**not connect** A and C" in this case you need an algorithm to disambuate your
    action.

    The section "examples" below provides a concrete example on what we mean by that and how to make it
    working.

    It can not yet be converted to / from gym space. If this feature is interesting for you, you can
    reply to the issue posted at https://github.com/rte-france/Grid2Op/issues/16

    **NB** compare to :class:`IdToAct` this converter allows for a smaller size. If you have N elements connected at
    a substation, you end up with `N*(N-1)/2` different action. Compare to IdToAct though, it is expected that your
    algorithm produces more than 1 outputs.

    **VERY IMPORTANT** : for this converter to work, it needs to remember the previous state of the grid, so you
    absolutely need to call its method :func:`ConnectivityConverter.convert_obs` a each observation.

    .. note:: This converter does not allow to affect the status (connected / disconnected) of the objects, neither
        to perform redispatching actions, neither to perform actions on storage units.

    Examples
    --------
    TODO: documentation in progress

    The idea of this converter is to allow to provide an interface if you want to provide action with what elements
    should be connected together.

    This is useful if an agent should reason on the target graph of the grid rather than reasoning on which
    elements are connected on which busbar.

    This converters then expects a vector of floats, all in [0., 1.]. The number of components of this vector
    is determined once and for all at the initialization and is accessible with `converter.n`. This is determined
    with the following rule. A pair of element of the grid el_i, el_j (elements here is: load, generator, storage
    unit, origin side of a powerline, extremity side of a powerline):

    - el_i and el_j belongs at the same substation
    - the substation to which el_i and el_j belongs counts 4 or more elements

    You can access which pair of elements is encoded for each component of this vector with
    :func:`ConnectivityConverter.which_pairs`.

    To create use the connectivity converter, you can:

    .. code-block:: python

        import grid2op
        import numpy as np
        from grid2op.Converter import ConnectivityConverter

        env = grid2op.make("l2rpn_case14_sandbox", test=True)
        converter = ConnectivityConverter(env.action_space)
        # it's a good practice to seed the element that can be, for reproducibility
        converter.seed(0)
        # to avoid creating illegal actions affecting more than the allowed number of parameters
        converter.init_converter(max_sub_changed=env.parameters.MAX_SUB_CHANGED)

    This converter is expected to receive a vector of the proper size with components being floats, representing:

    - -1.000...: the pairs should not be connected
    - 1.000...: the pairs should be connected
    - 0.000...: i have no opinion on this pairs of objects

    It uses an heuristic (greedy) to compute a resulting target topology (the vector `act.set_bus`) that tries to
    minimize the "disagreement" between the connectivity provided and the topology computed.

    More concretely, say you have 4 objects el1, el2, el3 and el4 connected on a substation. You want:

    - el1 connected to el2 with score of 0.7


    In the above example, we can change the connectivity of 77 pairs of elements, being:

    .. code-block:: python

        print(f"The connectivity of {converter.n} pairs of elements can be affected")
        for i in range(converter.n):
            sub_id, (type0, id0), (type1, id1) = converter.which_pairs(i)
            print(f"You can decide to connect / disconnect the \"{type0} id {id0}\" and the \"{type1} id {id1}\" at "
                  f"substation {sub_id} by action on component {i}")

    For example, if you want, at substation 1 to have:

    - "line_ex id 0", "line_or id 2" and "load id 0" on the same busbar
    - "line_or id 3", "line_or id 4" and "gen id 0" on the other one

    You can (this is one of the possible way to do it):

    .. code-block:: python

        encoded_act = np.zeros(converter.n)
        encoded_act[0] = 1  # i want to connect  "line_ex id 0" and the "line_or id 2"
        encoded_act[1] = -1  # i don't want to connect "line_ex id 0" and the "line_or id 3"
        encoded_act[2] = -1  # i don't want to connect "line_ex id 0" and the "line_or id 4"
        encoded_act[3] = -1  # i don't want to connect "line_ex id 0" and the "gen id 0"
        encoded_act[4] = 1  # i want to connect "line_ex id 0" and the "load id 0"

        # and now retrieve the corresponding grid2op action:
        grid2op_act = converter.convert_act(encoded_act)
        print(grid2op_act)

    Another one, to express exactly the same action:

    .. code-block:: python

        encoded_act2 = np.zeros(converter.n)
        encoded_act2[0] = 1  # i want to connect  "line_ex id 0" and the "line_or id 2"
        encoded_act2[4] = 1  # i want to connect "line_ex id 0" and the "load id 0"

        encoded_act2[9] = 1 # i want to connect "line_or id 3" and the "line_or id 4"
        encoded_act2[10] = 1  # i want to connect "line_or id 3" and the "gen id 0"

        encoded_act2[14] = -1  # i don't want to connect "gen id 0" and the "load id 0"

        # and now retrieve the corresponding grid2op action:
        grid2op_act2 = converter.convert_act(encoded_act2)
        print(grid2op_act2)

    In most cases, "something" (*eg* a neural network) is responsible to predict the "encoded action" and this
    converter can then be used to convert it to a valid grid2op action.

    Notes
    ------
    This converter does not allow to connect / disconnect any object. This feature might be added in the future.

    This converter takes as input a vector of (-1, 1) each component representing the "score" of the corresponding
    pairs of element on the grid to be connected or disconnected.

    A perfect converter would minimize (the variables are the component of `act.set_bus` vector that can
    be either 0 (i dont change) 1 or 2) the sum, for
    all index `i` fo pairs of elements in the grid el_k, el_j (that are encoded at position `i`)
    `1 - encoded_act[i]` if the pairs of elements el_k, el_j are on the same
    busbar {*i.e* iif (`act.set_bus[el_k] == 1` and `act.set_bus[el_j] == 1`) or
    (`act.set_bus[el_k] == 2` and `act.set_bus[el_j] == 2`)} and `1 + encoded_act[i]` otherwise
    {*i.e* iif (`act.set_bus[el_k] == 1` and `act.set_bus[el_j] == 2`) or
    (`act.set_bus[el_k] == 2` and `act.set_bus[el_j] == 1`)}.

    For now a heuristic based on a greedy approach is used. This is far from giving an "optimal" solution.

    This heuristic tries to act on as little elements as possible.

    """

    def __init__(self, action_space):
        if not action_space.supports_type("set_bus"):
            raise RuntimeError(
                "It is not possible to use the connectivity converter if the action space do not "
                'support the "set_bus" argument.'
            )

        Converter.__init__(self, action_space)
        self.__class__ = ConnectivityConverter.init_grid(action_space)
        self.subs_ids = np.array([], dtype=dt_int)
        self.obj_type = []
        self.pos_topo = np.array([], dtype=dt_int)

        self.n = -1
        self.last_obs = None
        self.max_sub_changed = self.n_sub
        self.last_disagreement = None
        self.indx_sel = None  # for explore in "convert_act"

    def init_converter(self, all_actions=None, **kwargs):
        # compute all pairs of elements that can be connected together
        self.pos_topo = []
        self.subs_ids = []
        for sub_id, nb_element in enumerate(self.sub_info):
            if nb_element < 4:
                continue

            c_id = np.nonzero(self.load_to_subid == sub_id)[0]
            g_id = np.nonzero(self.gen_to_subid == sub_id)[0]
            lor_id = np.nonzero(self.line_or_to_subid == sub_id)[0]
            lex_id = np.nonzero(self.line_ex_to_subid == sub_id)[0]
            storage_id = np.nonzero(self.storage_to_subid == sub_id)[0]

            c_pos = self.load_to_sub_pos[self.load_to_subid == sub_id]
            g_pos = self.gen_to_sub_pos[self.gen_to_subid == sub_id]
            lor_pos = self.line_or_to_sub_pos[self.line_or_to_subid == sub_id]
            lex_pos = self.line_ex_to_sub_pos[self.line_ex_to_subid == sub_id]
            storage_pos = self.storage_to_sub_pos[self.storage_to_subid == sub_id]

            my_types = []
            pos_topo = []
            next_load_ = 0
            next_gen_ = 0
            next_lor_ = 0
            next_lex_ = 0
            next_storage_ = 0
            next_load = c_id[next_load_] if c_id.shape[0] > 0 else None
            next_gen = g_id[next_gen_] if g_id.shape[0] > 0 else None
            next_lor = lor_id[next_lor_] if lor_id.shape[0] > 0 else None
            next_lex = lex_id[next_lex_] if lex_id.shape[0] > 0 else None
            next_storage = (
                storage_id[next_storage_] if storage_id.shape[0] > 0 else None
            )
            for id_i in range(nb_element):
                type_i, id_obj_i = self._get_id_from_obj(
                    id_i,
                    c_pos,
                    g_pos,
                    lor_pos,
                    lex_pos,
                    storage_pos,
                    next_load,
                    next_gen,
                    next_lor,
                    next_lex,
                    next_storage,
                )
                if type_i == "load":
                    next_load_ += 1
                    next_load = c_id[next_load_] if c_id.shape[0] > next_load_ else None
                elif type_i == "gen":
                    next_gen_ += 1
                    next_gen = g_id[next_gen_] if g_id.shape[0] > next_gen_ else None
                elif type_i == "line_or":
                    next_lor_ += 1
                    next_lor = (
                        lor_id[next_lor_] if lor_id.shape[0] > next_lor_ else None
                    )
                elif type_i == "line_ex":
                    next_lex_ += 1
                    next_lex = (
                        lex_id[next_lex_] if lex_id.shape[0] > next_lex_ else None
                    )
                elif type_i == "storage":
                    next_storage_ += 1
                    next_storage = (
                        storage_id[next_storage_]
                        if storage_id.shape[0] > next_storage_
                        else None
                    )
                else:
                    raise RuntimeError(f"Unsupported object type: {type_i}")
                my_types.append((type_i, id_obj_i))
                pos_topo.append(self._get_pos_topo(type_i, id_obj_i))

            for id_i in range(nb_element):
                id_i_ = my_types[id_i]
                pos_topo_i = pos_topo[id_i]
                for id_j in range(id_i + 1, nb_element):
                    id_j_ = my_types[id_j]
                    pos_topo_j = pos_topo[id_j]
                    self.obj_type.append((sub_id, id_i_, id_j_))
                    self.pos_topo.append((pos_topo_i, pos_topo_j))
                    self.subs_ids.append(sub_id)

        self.pos_topo = np.array(self.pos_topo)
        self.subs_ids = np.array(self.subs_ids)
        self.n = self.subs_ids.shape[0]

        if "max_sub_changed" in kwargs:
            self.max_sub_changed = int(kwargs["max_sub_changed"])

    def _get_id_from_obj(
        self,
        id_,
        c_pos,
        g_pos,
        lor_pos,
        lex_pos,
        storage_pos,
        next_load,
        next_gen,
        next_lor,
        next_lex,
        next_storage,
    ):
        if id_ in c_pos:
            type_ = "load"
            id_obj_ = next_load
        elif id_ in g_pos:
            type_ = "gen"
            id_obj_ = next_gen
        elif id_ in lor_pos:
            type_ = "line_or"
            id_obj_ = next_lor
        elif id_ in lex_pos:
            type_ = "line_ex"
            id_obj_ = next_lex
        elif id_ in storage_pos:
            type_ = "storage"
            id_obj_ = next_storage
        else:
            raise RuntimeError("Invalid grid")
        return type_, id_obj_

    def _get_pos_topo(self, type_, id_obj):
        if type_ == "load":
            res = self.load_pos_topo_vect[id_obj]
        elif type_ == "gen":
            res = self.gen_pos_topo_vect[id_obj]
        elif type_ == "line_or":
            res = self.line_or_pos_topo_vect[id_obj]
        elif type_ == "line_ex":
            res = self.line_ex_pos_topo_vect[id_obj]
        elif type_ == "storage":
            res = self.storage_pos_topo_vect[id_obj]
        else:
            raise RuntimeError("Invalid grid")
        return res

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

        **VERY IMPORTANT**: for this converter to work, it needs to remember the previous state of the grid, so you
        absolutely need to call its method :func:`ConnectivityConverter.convert_obs` at each observation.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The input observation.

        Returns
        -------

        transformed_obs: ``object``
            An different representation of the input observation, typically represented as a 1d vector that can be
            processed by a neural networks.

        """
        self.last_obs = obs
        return obs

    def convert_act(self, encoded_act, explore=None):
        """
        For this converter, encoded_act is a vector, with the same size as there are possible ways to reconfigure
        the grid.

        And it find a consistent state that does not break too much the connectivity asked.

        NOTE: there might be better ways to do it... This is computed with a greedy approach for now.

        Parameters
        ----------
        encoded_act: ``numpy.ndarray``
            This action should have the same size as the number of pairs of element that can be connected. A number
            close to -1 means you don't want to connect the pair together, a number close to +1 means you want the
            pairs to be connected together.

        explore: ``int``
            Defaults to ``None`` to be purely greedy. The higher `explore` the closer the returned solution will be to
            the "global optimum", but the longer it will takes. ``None`` will return the greedy approaches. Note that
            this is definitely not optimized for performance, and casting this problem into an optimization problem
            and solving this combinatorial optimization would definitely make this convereter more usefull.

        Returns
        -------
        act: :class:`grid2op.Action.BaseAction`
            The action that is usable by grid2op (after conversion) [the action space must be compatible with
            the "set_bus" key word]

        """
        encoded_act = np.array(encoded_act).astype(dt_float)
        if encoded_act.shape[0] != self.n:
            raise RuntimeError(
                f"Invalid encoded_act shape provided it should be {self.n}"
            )
        if ((encoded_act < -1.0) | (encoded_act > 1.0)).any():
            errors = (encoded_act < -1.0) | (encoded_act > 1.0)
            indexes = np.nonzero(errors)[0]
            raise RuntimeError(
                f'All elements of "encoded_act" must be in range [-1, 1]. Please check your '
                f"encoded action at positions {indexes[:5]}... (only first 5 displayed)"
            )

        act_want_change = np.abs(encoded_act) >= 1e-7
        encoded_act_filtered = encoded_act[act_want_change]
        if encoded_act_filtered.shape[0] == 0:
            # do nothing action in this case
            return super().__call__()

        argsort_changed = np.argsort(-np.abs(encoded_act_filtered))
        argsort = np.nonzero(act_want_change)[0][argsort_changed]
        act, disag = self._aux_act_from_order(argsort, encoded_act)
        self.indx_sel = 0
        if explore is None:
            pass
        elif isinstance(explore, int):
            # TODO better way here without a doubt! (combinatorial optimization, google OR-tools for example)
            for nb_exp in range(explore):
                # shuffle a bit the order i which i will built the action
                this_order = 1 * argsort
                self.space_prng.shuffle(this_order)
                # and now compute the action and the disagreement
                tmp_act, tmp_disag = self._aux_act_from_order(this_order, encoded_act)
                # if disagreement is lower than previous one, then take this action instead
                if tmp_disag < disag:
                    self.indx_sel = nb_exp + 1
                    act = tmp_act
                    disag = tmp_disag
        else:
            raise RuntimeError('Unknown parameters "explore" provided.')

        self.last_disagreement = disag
        return act

    def _aux_act_from_order(self, order, encoded_act):
        # TODO some part should be able to be vectorize i imagine
        topo_vect = np.zeros(self.dim_topo, dtype=dt_int)
        subs_added = np.full(self.n_sub, fill_value=False)
        sub_changed = 0
        order_id = (
            []
        )  # id of the pairs i have the right to modify (i can't always modifies everything due to
        # limit on self.max_sub_changed
        for el in order:
            my_sub = self.subs_ids[el]
            if not subs_added[my_sub]:
                if sub_changed < self.max_sub_changed:
                    subs_added[my_sub] = True
                    topo_vect[
                        self.pos_topo[el, 0]
                    ] = 1  # assign to +1 the first element of the substation met
                    sub_changed += 1
                    order_id.append(el)  # i need to modify this element later on:
                    # because it's the first element of a substation and i have the right to modify the substation
            else:
                # i need to modify this element later on:
                # because i modify its substation already.
                order_id.append(el)

        order = np.array(order_id)
        while order.shape[0] > 0:
            new_order = []
            for el in order:
                bus_1_id = self.pos_topo[el, 0]
                bus_2_id = self.pos_topo[el, 1]
                need_1 = topo_vect[bus_1_id] <= 0
                need_2 = topo_vect[bus_2_id] <= 0
                val = encoded_act[el]
                if need_2 and not need_1:
                    if val > 0.0:
                        # they are likely on same bus
                        topo_vect[bus_2_id] = topo_vect[bus_1_id]
                    elif val < 0.0:
                        # they are likely on different bus
                        topo_vect[bus_2_id] = 1 - topo_vect[bus_1_id] + 2
                elif need_1 and not need_2:
                    if val > 0.0:
                        # they are likely on same bus
                        topo_vect[bus_1_id] = topo_vect[bus_2_id]
                    elif val < 0.0:
                        # they are likely on different bus
                        topo_vect[bus_1_id] = 1 - topo_vect[bus_2_id] + 2
                elif need_1 and need_2:
                    # i don't have enough information yet to find a good placement for these
                    new_order.append(el)
            if set(new_order) == set(order):
                # i don't have constraints to solve the problem, i add something articially
                topo_vect[self.pos_topo[new_order[0], 0]] = 1
            order = np.array(new_order)

        act = super().__call__({"set_bus": topo_vect})
        dis_ = self._compute_disagreement(encoded_act, topo_vect)
        return act, dis_

    def _compute_disagreement(self, encoded_act, topo_vect):
        """
        INTERNAL

         .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Computes the disagreement between the encoded act and the proposed topo_vect

        **NB** if encoded act is random uniform, and topo_vect is full of 1, then disagreement is, on average 0.5.

        Lower disagreement is always better.
        """
        set_component = np.abs(encoded_act) >= 1e-7
        bus_el1 = topo_vect[self.pos_topo[:, 0]]
        bus_el2 = topo_vect[self.pos_topo[:, 1]]
        # for the element that will connected
        together = 1 - encoded_act[(bus_el1 == bus_el2) & (bus_el1 > 0) & set_component]
        # for the element that will be disconnected
        split = (
            1
            + encoded_act[
                (bus_el1 != bus_el2) & (bus_el1 > 0) & (bus_el2 > 0) & set_component
            ]
        )
        # for the elements that are not affected by the action (i don't know where they will be: maximum penalty)
        not_set = np.full(
            (((bus_el1 == 0) | (bus_el2 == 0)) & set_component).sum(),
            fill_value=2,
            dtype=dt_int,
        )

        # total disagreement
        raw_disag = together.sum() + split.sum() + not_set.sum()
        scaled_disag = raw_disag / self.n * 0.5  # to have something between 0 and 1
        return scaled_disag

    def sample(self):
        coded_act = self.space_prng.rand(self.n) * 2.0 - 1.0
        return self.convert_act(coded_act)

    def which_pairs(self, pair_id):
        """
        Returns a description of the pair of element that is encoded at position `pair_id` of the `encoded_act`

        Parameters
        ----------
        pair_id: ``int``


        Returns
        -------
        res: ``tuple``
            Tuple of 3 elements containing:

            - `sub_id` the id of the substation affected by the component `pair_id`
            - (obj_type, obj_id) the i
        """
        try:
            pair_id = int(pair_id)
        except Exception as exc_:
            raise RuntimeError(
                f'Invalid "pair_id" provided, it should be of integer type. Error was: \n"{exc_}"'
            )
        if pair_id < 0:
            raise RuntimeError(f'"pair_id" should be positive. You provided {pair_id}')
        if pair_id >= self.n:
            raise RuntimeError(
                f'"pair_id" should be lower than the size of the action space, in this case '
                f"{self.n}. You provided {pair_id}"
            )
        return self.obj_type[pair_id]

    def do_nothing_encoded_act(self):
        """returns the do nothing encoding act"""
        return np.zeros(self.n, dtype=dt_float)
