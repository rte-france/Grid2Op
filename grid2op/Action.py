"""
The "Action" module lets you define some actions on the underlying power _grid.
These actions are either made by an agent, or by the environment.

For now, the actions can act on:

  - the "injections" and allows you to change:

    - the generators active power production setpoint
    - the generators voltage magnitude setpoint
    - the loads active power consumption
    - the loads reactive power consumption

  - the status of the powerlines (connected/disconnected)
  - the configuration at substations eg setting different objects to different buses for example

The Action class is abstract. You can implement it the way you want. If you decide to extend it, make sure
that the :class:`grid2op.Backend` class will be able to understand it. If you don't, your extension will not affect the
underlying powergrid. Indeed a :class:`grid2op.Backend` will call the :func:`Action.__call__` method and should
understands its return type.

In this module we derived two action class:

  - :class:`Action` represents a type of action that can act on all the above-mentioned objects
  - :class:`TopologyAction` restricts the modification to line status modification and bus reconfiguration at substations.


The :class:`Action` and all its derivatives also offer some usefull inspection utilities:

  - :func:`Action.__str__` prints the action in a format that gives usefull information on how it will affect the powergrid
  - :func:`Action.effect_on` returns a dictionnary that gives information about its effect.

Finally, :class:`Action` class define some strict behavior to follow if reimplementing them. The correctness of each
instances of Action is assessed both when calling :func:`Action.update` or with a call to
:func:`Action._check_for_ambiguity` performed for example by the Backend when it must implement its effect on the
powergrid through a call to :func:`Action.__call__`

"""


import numpy as np
import warnings
import itertools

import pdb

try:
    from .Exceptions import *
    from .Space import SerializableSpace, GridObjects
except (ModuleNotFoundError, ImportError):
    from Exceptions import *
    from Space import SerializableSpace, GridObjects


# TODO code "reduce" multiple action (eg __add__ method, carefull with that... for example "change", then "set" is not
# ambiguous at all, same with "set" then "change")


# TODO code "convert_for" and "convert_from" to be able to change the backend (should be handled by the backend directly)
# TODO have something that output a dict like "i want to change this element" (with a simpler API than the update stuff)
# TODO time delay somewhere (eg action is implemented after xxx timestep, and not at the time where it's proposed)

# TODO have the "reverse" action, that does the opposite of an action. Will be hard but who know ? :eyes:

# TODO tests for redispatching action.

class Action(GridObjects):
    """
    This is a base class for each :class:`Action` objects.
    As stated above, an action represents conveniently the modifications that will affect a powergrid.

    It is not recommended to instantiate an action from scratch. The recommended way to get an action is either by
    modifying an existing one using the method :func:`Action.update` or to call and :class:`HelperAction` object that
    has been properly set up by an :class:`grid2op.Environment`.

    Action can be fully converted to and back from a numpy array with a **fixed** size.

    An action can modify the grid in multiple ways.
    It can change :

    - the production and voltage setpoint of the generator units
    - the amount of power consumed (for both active and reactive part) for load
    - disconnect powerlines
    - change the topology of the _grid.

    To be valid, an action should be convertible to a tuple of 5 elements:

    - the first element is the "injections" vector: representing the way generator units and loads are modified
        - It is, in turn, a dictionary with the following keys (optional)

            - "load_p" a vector of the same size of the load, giving the modification of the loads active consumption
            - "load_q" a vector of the same size of the load, giving the modification of the loads reactive consumption
            - "prod_p" a vector of the same size of the generators, giving the modification of the productions active
              setpoint production
            - "prod_v" a vector of the same size of the generators, giving the modification of the productions voltage
              setpoint

    - the second element is made of force line status. It is made of a vector of size :attr:`Action._n_lines`
      (the number of lines in the powergrid) and is interpreted as:

            - -1 force line disconnection
            - +1 force line reconnection
            - 0 do nothing to this line

    - the third element is the switch line status vector. It is made of a vector of size :attr:`Action._n_lines` and is
      interpreted as:

        - ``True``: change the line status
        - ``False``: don't do anything

    - the fourth element set the buses to which the object is connected. It's a vector of integers with the following
      interpretation:

        - 0 -> don't change
        - 1 -> connect to bus 1
        - 2 -> connect to bus 2
        - -1 -> disconnect the object.

    - the fifth element changes the buses to which the object is connected. It's a boolean vector interpreted as:
        - ``False``: nothing is done
        - ``True``: change the bus eg connect it to bus 1 if it was connected to bus 2 or connect it to bus 2 if it was
          connected to bus 1. NB this is only active if the system has only 2 buses per substation (that's the case for
          the L2RPN challenge).

    - the sixth element is a vector, representing the redispatching. Component of this vector is added to the
      generators active setpoint value (if set) of the first elements.

    **NB** the difference between :attr:`Action._set_topo_vect` and :attr:`Action._change_bus_vect` is the following:

        - If  a component of :attr:`Action._set_topo_vect` is 1, then the object (load, generator or powerline)
          will be moved to bus 1 of the substation to which it is connected. If it is already to bus 1 nothing will be
          done.
          If it's on another bus it will connect it to bus 1. It's disconnected, it will reconnect it and connect it
          to bus 1.
        - If a component of :attr:`Action._change_bus_vect` is True, then the object will be moved from one bus to
          another.
          If the object were on bus 1
          it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
          disconnected,
          then this does nothing.

    The conversion to the action into an understandable format by the backend is performed by the "update" method,
    that takes into account a dictionary and is responsible to convert it into this format.
    It is possible to overload this class as long as the overloaded :func:`Action.__call__` operator returns the
    specified format, and the :func:`Action.__init__` method has the same signature.

    This format is then digested by the backend and the powergrid is modified accordingly.

    Attributes
    ----------

    _set_line_status: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the effect of the action on the status of it. It should be understood as:

          - -1: disconnect the powerline
          - 0: don't affect the powerline
          - +1: reconnect the powerline

    _switch_line_status: :class:`numpy.ndarray`, dtype:bool
        For each powerline, it informs whether the action will switch the status of a powerline of not. It should be
        understood as followed:

          - ``False``: the action doesn't affect the powerline
          - ``True``: the action affects the powerline. If it was connected, it will disconnect it. If it was
            disconnected, it will reconnect it.

    _dict_inj: ``dict``
        Represents the modification of the injection (productions and loads) of the power _grid. This dictionary can
        have the optional keys:

            - "load_p" to set the active load values (this is a numpy array with the same size as the number of load
              in the power _grid with Nan: don't change anything, else set the value
            - "load_q": same as above but for the load reactive values
            - "prod_p": same as above but for the generator active setpoint values. It has the size corresponding
              to the number of generators in the test case.
            - "prod_v": same as above but set the voltage setpoint of generator units.

    _set_topo_vect: :class:`numpy.ndarray`, dtype:int
        Similar to :attr:`Action._set_line_status` but instead of affecting the status of powerlines, it affects the
        bus connectivity at a substation. It has the same size as the full topological vector (:attr:`Action._dim_topo`)
        and for each element it should be understood as:

            - 0: nothing is changed for this element
            - +1: this element is affected to bus 1
            - -1: this element is affected to bus 2

    _change_bus_vect: :class:`numpy.ndarray`, dtype:bool
         Similar to :attr:`Action._switch_line_status` but it affects the topology at substations instead of the status
         of
         the powerline. It has the same size as the full topological vector (:attr:`Action._dim_topo`) and each
         component should mean:

             - ``False``: the object is not affected
             - ``True``: the object will be moved to another bus. If it was on bus 1 it will be moved on bus 2, and if
               it was on bus 2 it will be moved on bus 1.

    authorized_keys: :class:`set`
        The set indicating which keys the actions can understand when calling :func:`Action.update`

    _subs_impacted: :class:`numpy.ndarray`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each substation, if it is impacted
        by the action (in this case :attr:`Action._subs_impacted`\[sub_id\] is ``True``) or not
        (in this case :attr:`Action._subs_impacted`\[sub_id\] is ``False``)

    _lines_impacted: :class:`numpy.ndarray`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each powerline, if it is impacted
        by the action (in this case :attr:`Action._lines_impacted`\[line_id\] is ``True``) or not
        (in this case :attr:`Action._subs_impacted`\[line_id\] is ``False``)

    vars_action: ``list``, static
        The authorized key that are processed by :func:`Action.__call__` to modify the injections

    vars_action_set: ``set``, static
        The authorized key that is processed by :func:`Action.__call__` to modify the injections

    _redispatch: :class:`numpy.ndarray`, dtype:float
        Amount of redispatching that this action will perform. Redispatching will increase the generator's active
        setpoint
        value. This will be added to the value of the generators. The Environment will make sure that every physical
        constraint is met. This means that the agent provides a setpoint, but there is no guarantee that the setpoint
        will be achievable. Redispatching action is cumulative, this means that if at a given timestep you ask +10 MW
        on a generator, and on another you ask +10 MW then the total setpoint for this generator that the environment
        will try to implement is +20MW.

    """

    vars_action = ["load_p", "load_q", "prod_p", "prod_v"]
    vars_action_set = set(vars_action)

    def __init__(self, gridobj):
        """
        This is used to create an Action instance. Preferably, :class:`Action` should be created with
        :class:`HelperAction`.

        **It is NOT recommended** to create an action with this method. Please use :func:`HelperAction.__call__` or
        :func:`HelperAction.sample` to create a valid action.

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the objects present in the powergrid

        """
        GridObjects.__init__(self)
        self.init_grid(gridobj)

        self.authorized_keys = {"injection",
                                "hazards", "maintenance", "set_line_status", "change_line_status",
                                "set_bus", "change_bus", "redispatch"}

        # False(line is disconnected) / True(line is connected)
        self._set_line_status = None
        self._switch_line_status = None

        # injection change
        self._dict_inj = {}

        # redispatching
        self._redispatch = None

        # topology changed
        self._set_topo_vect = None
        self._change_bus_vect = None

        self._vectorized = None

        self._subs_impacted = None
        self._lines_impacted = None

        # add the hazards and maintenance usefull for saving.
        self._hazards = None
        self._maintenance = None

        self.reset()

        # decomposition of the Action into homogeneous sub-spaces
        self.attr_list_vect = ["prod_p", "prod_v", "load_p", "load_q", "_redispatch",
                               "_set_line_status", "_switch_line_status",
                               "_set_topo_vect", "_change_bus_vect", "_hazards", "_maintenance"]

        self._single_act = True

    def _get_array_from_attr_name(self, attr_name):
        if attr_name in self.__dict__:
            res = super()._get_array_from_attr_name(attr_name)
        else:
            if attr_name in self._dict_inj:
                res = self._dict_inj[attr_name]
            else:
                if attr_name == "prod_p" or attr_name == "prod_v":
                    res = np.full(self.n_gen, fill_value=0., dtype=np.float)
                elif attr_name == "load_p" or attr_name == "load_q":
                    res = np.full(self.n_load, fill_value=0., dtype=np.float)
                else:
                    raise Grid2OpException("Impossible to find the attribute \"{}\" "
                                           "into the Action of type \"{}\"".format(attr_name, type(self)))
        return res

    def _assign_attr_from_name(self, attr_nm, vect):
        if attr_nm in self.__dict__:
            super()._assign_attr_from_name(attr_nm, vect)
        else:
            if np.any(np.isfinite(vect)):
                if np.any(vect != 0.):
                    self._dict_inj[attr_nm] = vect

    def check_space_legit(self):
        """
        This method allows to check if this method is ambiguous **per se** (defined more formally as:
        whatever the observation at time *t*, and the changes that can occur between *t* and *t+1*, this
        action will be ambiguous).

        For example, an action that try to assign something to busbar 3 will be ambiguous *per se*. An action
        that tries to dispatch a non dispatchable generator will be also ambiguous *per se*.

        However, an action that "switch" (change) the status (connected / disconnected) of a powerline can be
        ambiguous and it will not be detected here. This is because the ambiguity depends on the current state
        of the powerline:

        - if the powerline is disconnected, changing its status means reconnecting it. And we cannot reconnect a
          powerline without specifying on which bus.
        - on the contrary if the powerline is connected, changing its status means disconnecting it, which is
          always feasible.

        In case of "switch" as we see here, the action can be ambiguous, but not ambiguous *per se*. This method
        will **never** throw any error in this case.

        Returns
        -------

        """
        self._check_for_ambiguity()

    def get_set_line_status_vect(self):
        """
        Computes and returns a vector that can be used in the :func:`Action.__call__` with the keyword
        "set_status" if building an :class:`Action`.

        **NB** this vector is not the internal vector of this action but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.int
            A vector that doesn't affect the grid, but can be used in :func:`Action.__call__` with the keyword
            "set_status" if building an :class:`Action`.

        """
        return np.full(shape=self.n_line, fill_value=0, dtype=np.int)

    def get_change_line_status_vect(self):
        """
        Computes and returns a vector that can be used in the :func:`Action.__call__` with the keyword
        "set_status" if building an :class:`Action`.

        **NB** this vector is not the internal vector of this action but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.bool
            A vector that doesn't affect the grid, but can be used in :func:`Action.__call__` with the keyword
            "set_status" if building an :class:`Action`.

        """
        return np.full(shape=self.n_line, fill_value=False, dtype=np.bool)

    def __eq__(self, other) -> bool:
        """
        Test the equality of two actions.

        2 actions are said to be identical if they have the same impact on the powergrid. This is unrelated to their
        respective class. For example, if an Action is of class :class:`Action` and doesn't act on the injection, it
        can be equal to an Action of the derived class :class:`TopologyAction` (if the topological modifications are the
        same of course).

        This implies that the attributes :attr:`Action.authorized_keys` is not checked in this method.

        Note that if 2 actions don't act on the same powergrid, or on the same backend (eg number of loads, or
        generators are not the same in *self* and *other*, or they are not in the same order) then action will be
        declared as different.

        **Known issue** if two backends are different, but the description of the _grid are identical (ie all
        n_gen, n_load, n_line, sub_info, dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behavior: if everything is the same, then probably, up to
        the naming convention, then the power grids are identical too.

        Parameters
        ----------
        other: :class:`Action`
            An instance of class Action to which "self" will be compared.

        Returns
        -------
        res: ``bool``
            Whether the actions are equal or not.

        """

        # check that the _grid is the same in both instances
        same_grid = True
        same_grid = same_grid and self.n_gen == other.n_gen
        same_grid = same_grid and self.n_load == other.n_load
        same_grid = same_grid and self.n_line == other.n_line
        same_grid = same_grid and np.all(self.sub_info == other.sub_info)
        same_grid = same_grid and self.dim_topo == other.dim_topo
        # to which substation is connected each element
        same_grid = same_grid and np.all(self.load_to_subid == other.load_to_subid)
        same_grid = same_grid and np.all(self.gen_to_subid == other.gen_to_subid)
        same_grid = same_grid and np.all(self.line_or_to_subid == other.line_or_to_subid)
        same_grid = same_grid and np.all(self.line_ex_to_subid == other.line_ex_to_subid)
        # which index has this element in the substation vector
        same_grid = same_grid and np.all(self.load_to_sub_pos == other.load_to_sub_pos)
        same_grid = same_grid and np.all(self.gen_to_sub_pos == other.gen_to_sub_pos)
        same_grid = same_grid and np.all(self.line_or_to_sub_pos == other.line_or_to_sub_pos)
        same_grid = same_grid and np.all(self.line_ex_to_sub_pos == other.line_ex_to_sub_pos)
        # which index has this element in the topology vector
        same_grid = same_grid and np.all(self.load_pos_topo_vect == other.load_pos_topo_vect)
        same_grid = same_grid and np.all(self.gen_pos_topo_vect == other.gen_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_or_pos_topo_vect == other.line_or_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_ex_pos_topo_vect == other.line_ex_pos_topo_vect)
        if not same_grid:
            return False

        # _grid is the same, now I test the the injections modifications are the same
        same_action = self._dict_inj.keys() == other._dict_inj.keys()
        if not same_action:
            return False

        # all injections are the same
        for el in self._dict_inj.keys():
            if not np.all(self._dict_inj[el] == other._dict_inj[el]):
                return False

        # same line status
        if not np.all(self._set_line_status == other._set_line_status):
            return False
        if not np.all(self._switch_line_status == other._switch_line_status):
            return False

        # redispatching is same
        if not np.all(self._redispatch == other._redispatch):
            return False

        # same topology changes
        if not np.all(self._set_topo_vect == other._set_topo_vect):
            return False
        if not np.all(self._change_bus_vect == other._change_bus_vect):
            return False

        # objects are the same
        return True

    def get_topological_impact(self):
        """
        Gives information about the element being impacted by this action.

        **NB** The impacted elements can be used by :class:`grid2op.LegalAction` to determine whether or not an action
        is legal or not.

        **NB** The impacted are the elements that can potentially be impacted by the action. This does not mean they
        will be impacted. For examples:

            - If an action from an :class:`grid2op.Agent` reconnect a powerline, but this powerline is being
              disconnected by a hazard at the same time step, then this action will not be implemented on the grid.
              However, it this powerline couldn't be reconnected for some reason (for example it was already out of
              order) the action will still be declared illegal, even if it has NOT impacted the powergrid.
            - If an action tries to disconnect a powerline already disconnected, it will "impact" this powergrid. This
              means that even if the action will do nothing, it disconnecting this powerline is against the
              rules, then the action will be illegal.
            - If an action tries to change the topology of a substation, but this substation is already at the target
              topology, the same mechanism applies. The action will "impact" the substation, even if, in the end, it
              consists of doing nothing.

        Any such "change" that would be illegal is declared as "illegal" regardless of the real impact of this action
        on the powergrid.


        Returns
        -------
        lines_impacted: :class:`numpy.array`, dtype:np.bool
            A vector with the same size as the number of powerlines in the grid (:attr:`Action.n_line`) with for each
            component ``True`` if the line STATUS is impacted by the action, and ``False`` otherwise. See
            :attr:`Action._lines_impacted` for more information.

        subs_impacted: :class:`numpy.array`, dtype:np.bool
            A vector with the same size as the number of substations in the grid with for each
            component ``True`` if the substation is impacted by the action, and ``False`` otherwise. See
            :attr:`Action._subs_impacted` for more information.

        """
        if self._lines_impacted is None:
            self._lines_impacted = self._switch_line_status | (self._set_line_status != 0)

        if self._subs_impacted is None:
            # supposes tha self._lines_impacted
            self._subs_impacted = np.full(shape=self.sub_info.shape, fill_value=False, dtype=np.bool)
            beg_ = 0
            end_ = 0
            powerlines_reco = np.where(self._set_line_status == 1)[0]  # all the id of the powerlines reconnected
            sub_or_id = self.line_or_to_subid[powerlines_reco]
            sub_ex_id = self.line_ex_to_subid[powerlines_reco]
            sub_id = np.concatenate((sub_or_id, sub_ex_id))
            sub_id_unique, sub_counts = np.unique(sub_id, return_counts=True)
            sub_counts = dict(zip(sub_id_unique, sub_counts))
            is_sub_concerned = np.full(shape=self.sub_info.shape, fill_value=False, dtype=np.bool)
            is_sub_concerned[sub_id_unique] = True
            for sub_id, nb_obj in enumerate(self.sub_info):
                nb_obj = int(nb_obj)
                end_ += nb_obj
                if np.any(self._change_bus_vect[beg_:end_]):
                    # change always impact the substations
                    self._subs_impacted[sub_id] = True
                nb_set = np.sum(self._set_topo_vect[beg_:end_] != 0)
                if nb_set > 0:
                    # if a powerline has been reconnected, don't count busor and busex as "impacted" if the action
                    # concerned only the reconnected powerline
                    # in some cases, set does not impact it then.
                    if not is_sub_concerned[sub_id]:
                        # no powerline are connected here so
                        self._subs_impacted[sub_id] = True
                    else:
                        # in this case, i reconnected a powerline having one of its end on a substation, so you might not
                        # need to count this action
                        if sub_counts[sub_id] != nb_set:
                            # in this case, only actions regarding reconnection of powerlines are performed
                            self._subs_impacted[sub_id] = True

                beg_ += nb_obj

        return self._lines_impacted, self._subs_impacted

    def reset(self):
        """
        Reset the action to the "do nothing" state.

        Returns
        -------

        """
        # False(line is disconnected) / True(line is connected)
        self._set_line_status = np.full(shape=self.n_line, fill_value=0, dtype=np.int)
        self._switch_line_status = np.full(shape=self.n_line, fill_value=False, dtype=np.bool)

        # injection change
        self._dict_inj = {}

        # topology changed
        self._set_topo_vect = np.full(shape=self.dim_topo, fill_value=0, dtype=np.int)
        self._change_bus_vect = np.full(shape=self.dim_topo, fill_value=False, dtype=np.bool)

        # add the hazards and maintenance usefull for saving.
        self._hazards = np.full(shape=self.n_line, fill_value=False, dtype=np.bool)
        self._maintenance = np.full(shape=self.n_line, fill_value=False, dtype=np.bool)

        # redispatching vector
        self._redispatch = np.full(shape=self.n_gen, fill_value=0., dtype=np.float)

        self._vectorized = None
        self._lines_impacted = None
        self._subs_impacted = None

    def __iadd__(self, other):
        """
        Add an action to this one.
        Adding an action to myself is equivalent to perform myself, and then perform other.

        Add will have the following properties:

            - it erase the previous changes to injections
            -

        Parameters
        ----------
        other: :class:`Action`

        Returns
        -------

        """

        # deal with injections
        for el in self.vars_action:
            if el in other._dict_inj:
                if el not in self._dict_inj:
                    self._dict_inj[el] = other._dict_inj[el]
                else:
                    val = other._dict_inj[el]
                    ok_ind = np.isfinite(val)
                    self._dict_inj[el][ok_ind] = val[ok_ind]

        # redispatching
        redispatching = other._redispatch
        if np.any(redispatching != 0.):
            ok_ind = np.isfinite(redispatching)
            self._redispatch[ok_ind] += redispatching[ok_ind]

        # set and change status
        other_set = other._set_line_status
        other_change = other._switch_line_status
        me_set = self._set_line_status
        me_change = self._switch_line_status

        # i set, but the other change, so it's equivalent to setting to the opposite
        # so change +1 becomes -1 and -1 becomes +1
        me_set[other_change] *= -1
        # i set, the other set
        me_set[other_set != 0] = other_set[other_set != 0]
        # i change, but so does the other, i do nothing
        me_change[other_change] = False
        # i change, but the other set, it's erased
        me_change[other_set != 0] = False
        self._set_line_status = me_set
        self._switch_line_status = me_change

        # set and change bus
        other_set = other._set_topo_vect
        other_change = other._change_bus_vect
        me_set = self._set_topo_vect
        me_change = self._change_bus_vect

        # i set, but the other change, so it's equivalent to setting to the opposite
        # so change +1 becomes +2 and +2 becomes +1
        me_set[other_change] -= 1  # 1 becomes 0 and 2 becomes 1
        me_set[other_change] *= -1  # 1 is 0 and 2 becomes -1
        me_set[other_change] += 2  # 1 is 2 and 2 becomes 1

        # i set, the other set
        me_set[other_set != 0] = other_set[other_set != 0]
        # i change, but so does the other, i do nothing
        me_change[other_change] = False
        # i change, but the other set, it's erased
        me_change[other_set != 0] = False
        self._set_topo_vect = me_set
        self._change_bus_vect = me_change
        return self

    def __call__(self):
        """
        This method is used to return the effect of the current action in a format understandable by the backend.
        This format is detailed below.

        This function must also integrate the redispatching strategy for the Action.

        It also performs a check of whether or not an action is "Ambiguous", eg an action that reconnect a powerline
        but doesn't specify on which bus to reconnect it is said to be ambiguous.

        If this :func:`Action.__call__` is overloaded, the call of :func:`Action._check_for_ambiguity` must be ensured
        by this the derived class.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is :attr:`Action._dict_inj`

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._switch_line_status`

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_topo_vect`

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            This array, that has the same size as the number of generators indicates for each generator the amount of
            redispatching performed by the action.

        Raises
        -------
        :class:`grid2op.Exceptions.AmbiguousAction`
            Or one of its derivate class.

        """

        self._check_for_ambiguity()

        dict_inj = self._dict_inj
        set_line_status = self._set_line_status
        switch_line_status = self._switch_line_status
        set_topo_vect = self._set_topo_vect
        change_bus_vect = self._change_bus_vect
        redispatch = self._redispatch

        return dict_inj, set_line_status, switch_line_status, set_topo_vect, change_bus_vect, redispatch

    def _digest_injection(self, dict_):
        # I update the action
        if "injection" in dict_:
            if dict_["injection"] is not None:
                tmp_d = dict_["injection"]
                for k in tmp_d:  # ["load_p", "prod_p", "load_q", "prod_v"]:
                    if k in self.vars_action_set:
                        self._dict_inj[k] = np.array(tmp_d[k])
                    else:
                        warn = "The key {} is not recognized by Action when trying to modify the injections.".format(k)
                        warnings.warn(warn)

    def _digest_setbus(self, dict_):
        if "set_bus" in dict_:
            if isinstance(dict_["set_bus"], np.ndarray):
                # complete nodal topology vector is already provided
                self._set_topo_vect = dict_["set_bus"]
            elif isinstance(dict_["set_bus"], dict):
                ddict_ = dict_["set_bus"]
                handled = False
                # authorized_keys = {"loads_id", "generators_id", "lines_or_id", "lines_ex_id", "substations_id"}
                if "loads_id" in ddict_:
                    tmp = ddict_["loads_id"]
                    handled = True
                    for (c_id, bus) in tmp:
                        if c_id >= self.n_line:
                            raise AmbiguousAction("Load {} doesn't exist".format(c_id))
                        self._set_topo_vect[self.load_pos_topo_vect[c_id]] = bus
                        # print("self.load_pos_topo_vect[l_id] {}".format(self.load_pos_topo_vect[l_id]))
                if "generators_id" in ddict_:
                    tmp = ddict_["generators_id"]
                    handled = True
                    for (g_id, bus) in tmp:
                        if g_id >= self.n_gen:
                            raise AmbiguousAction("Generator {} doesn't exist".format(g_id))
                        self._set_topo_vect[self.gen_pos_topo_vect[g_id]] = bus
                if "lines_or_id" in ddict_:
                    tmp = ddict_["lines_or_id"]
                    handled = True
                    for (l_id, bus) in tmp:
                        if l_id >= self.n_line:
                            raise AmbiguousAction("Powerline {} doesn't exist".format(l_id))
                        self._set_topo_vect[self.line_or_pos_topo_vect[l_id]] = bus
                if "lines_ex_id" in ddict_:
                    tmp = ddict_["lines_ex_id"]
                    handled = True
                    for (l_id, bus) in tmp:
                        if l_id >= self.n_line:
                            raise AmbiguousAction("Powerline {} doesn't exist".format(l_id))
                        self._set_topo_vect[self.line_ex_pos_topo_vect[l_id]] = bus
                if "substations_id" in ddict_:
                    handled = True
                    tmp = ddict_["substations_id"]
                    for (s_id, arr) in tmp:
                        if s_id >= self.sub_info.shape[0]:
                            raise AmbiguousAction("Substation {} doesn't exist".format(s_id))

                        s_id = int(s_id)
                        beg_ = int(np.sum(self.sub_info[:s_id]))
                        end_ = int(beg_ + self.sub_info[s_id])
                        self._set_topo_vect[beg_:end_] = arr
                if not handled:
                    msg = "Invalid way to set the topology. When dict_[\"set_bus\"] is a dictionnary it should have"
                    msg += " at least one of \"loads_id\", \"generators_id\", \"lines_or_id\", "
                    msg += "\"lines_ex_id\" or \"substations_id\""
                    msg += " as keys. None where found. Current used keys are: "
                    msg += "{}".format(sorted(ddict_.keys()))
                    raise AmbiguousAction(msg)
            elif dict_["set_bus"] is None:
                pass
            else:
                raise AmbiguousAction(
                    "Invalid way to set the topology. dict_[\"set_bus\"] should be a numpy array or a dictionnary.")

    def _digest_change_bus(self, dict_):
        if "change_bus" in dict_:
            if isinstance(dict_["change_bus"], np.ndarray):
                # topology vector is already provided
                self._change_bus_vect = dict_["change_bus"]
            elif isinstance(dict_["change_bus"], dict):
                ddict_ = dict_["change_bus"]
                if "loads_id" in ddict_:
                    tmp = ddict_["loads_id"]
                    for l_id in tmp:
                        self._change_bus_vect[self.load_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self.load_pos_topo_vect[l_id]]
                if "generators_id" in ddict_:
                    tmp = ddict_["generators_id"]
                    for g_id in tmp:
                        self._change_bus_vect[self.gen_pos_topo_vect[g_id]] = not self._change_bus_vect[
                            self.gen_pos_topo_vect[g_id]]
                if "lines_or_id" in ddict_:
                    tmp = ddict_["lines_or_id"]
                    for l_id in tmp:
                        self._change_bus_vect[self.line_or_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self.line_or_pos_topo_vect[l_id]]
                if "lines_ex_id" in ddict_:
                    tmp = ddict_["lines_ex_id"]
                    for l_id in tmp:
                        self._change_bus_vect[self.line_ex_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self.line_ex_pos_topo_vect[l_id]]
                if "substations_id" in ddict_:
                    tmp = ddict_["substations_id"]
                    for (s_id, arr) in tmp:
                        s_id = int(s_id)
                        beg_ = int(np.sum(self.sub_info[:s_id]))
                        end_ = int(beg_ + self.sub_info[s_id])
                        self._change_bus_vect[beg_:end_][arr] = ~self._change_bus_vect[beg_:end_][arr]
            elif dict_["change_bus"] is None:
                pass
            else:
                raise AmbiguousAction(
                    "Invalid way to set the topology. dict_[\"change_bus\"] should be a numpy array or a dictionnary.")

    def _digest_set_status(self, dict_):
        if "set_line_status" in dict_:
            # the action will disconnect a powerline
            # note that if a powerline is already disconnected, it does nothing
            # this action can both disconnect or reconnect a powerlines
            if isinstance(dict_["set_line_status"], np.ndarray):
                if dict_["set_line_status"] is not None:
                    if len(dict_["set_line_status"]) != self.n_line:
                        raise InvalidNumberOfLines(
                            "This \"set_line_status\" action acts on {} lines while there are {} in the grid".format(
                                len(dict_["set_line_status"]), self.n_line))
                    sel_ = dict_["set_line_status"] != 0

                    # update the line status vector
                    self._set_line_status[sel_] = dict_["set_line_status"][sel_].astype(np.int)
            else:
                for l_id, status_ in dict_["set_line_status"]:
                    self._set_line_status[l_id] = status_

    def _digest_hazards(self, dict_):
        if "hazards" in dict_:
            # set the values of the power lines to "disconnected" for element being "False"
            # does nothing to the others
            # an hazard will never reconnect a powerline
            if dict_["hazards"] is not None:
                tmp = dict_["hazards"]
                try:
                    tmp = np.array(tmp)
                except:
                    raise AmbiguousAction(
                        "You ask to perform hazard on powerlines, this can only be done if \"hazards\" is castable into a numpy ndarray")
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self.n_line:
                        raise InvalidNumberOfLines(
                            "This \"hazards\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self.n_line))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction("You can only ask hazards with int or boolean numpy array vector.")

                self._set_line_status[tmp] = -1
                self._hazards[tmp] = True
                # force ignore of any topological actions
                self._ignore_topo_action_if_disconnection(tmp)

    def _digest_maintenance(self, dict_):
        if "maintenance" in dict_:
            # set the values of the power lines to "disconnected" for element being "False"
            # does nothing to the others
            # a _maintenance operation will never reconnect a powerline
            if dict_["maintenance"] is not None:
                tmp = dict_["maintenance"]
                try:
                    tmp = np.array(tmp)
                except:
                    raise AmbiguousAction(
                        "You ask to perform maintenance on powerlines, this can only be done if \"maintenance\" is castable into a numpy ndarray")
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self.n_line:
                        raise InvalidNumberOfLines(
                            "This \"maintenance\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self.n_line))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction(
                        "You can only ask to perform lines maintenance with int or boolean numpy array vector.")
                self._set_line_status[tmp] = -1
                self._maintenance[tmp] = True
                self._ignore_topo_action_if_disconnection(tmp)

    def _digest_change_status(self, dict_):
        if "change_line_status" in dict_:
            # the action will switch the status of the powerline
            # for each element equal to 1 in this dict_["change_line_status"]
            # if the status is "disconnected" it will be transformed into "connected"
            # and if the status is "connected" it will be switched to "disconnected"
            # Lines with "0" in this vector are not impacted.
            if dict_["change_line_status"] is not None:
                tmp = dict_["change_line_status"]
                try:
                    tmp = np.array(tmp)
                except:
                    raise AmbiguousAction(
                        "You ask to change the bus status, this can only be done if \"change_status\" is castable into a numpy ndarray")
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self.n_line:
                        raise InvalidNumberOfLines(
                            "This \"change_line_status\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self.n_line))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction("You can only change line status with int or boolean numpy array vector.")
                self._switch_line_status[dict_["change_line_status"]] = True

    def __convert_and_redispatch(self, kk, val):
        try:
            kk = int(kk)
            val = float(val)
        except Exception as e:
            raise AmbiguousAction("In redispatching, it's not possible to understand the key/value pair "
                                  "{}/{} provided in the dictionnary. Key must be an integer, value "
                                  "a float".format(kk, val))
        self._redispatch[kk] = val

    def _digest_redispatching(self, dict_):
        if "redispatch" in dict_:
            if dict_["redispatch"] is None:
                return
            tmp = dict_["redispatch"]
            if isinstance(tmp, np.ndarray):
                # complete redispatching is provided
                self._redispatch = tmp
            elif isinstance(tmp, dict):
                # dict must have key: generator to modify, value: the delta value applied to this generator
                ddict_ = tmp
                for kk, val in ddict_.items():
                    kk, val = self.__convert_and_redispatch(kk, val)
            elif isinstance(tmp, list):
                # list of tuples: each tupe (k,v) being the same as the key/value describe above
                treated = False
                if len(tmp) == 2:
                    if isinstance(tmp[0], tuple):
                        # there are 2 tuples in the list, i dont treat it as a tuple
                        treated = False
                    else:
                        # i treat it as a tuple
                        if len(tmp) != 2:
                            raise AmbiguousAction("When asking for redispatching with a tuple, you should make a "
                                                  "of tuple of 2 elements, the first one being the id of the"
                                                  "generator to redispatch, the second one the value of the "
                                                  "redispatching.")
                        kk, val = tmp
                        self.__convert_and_redispatch(kk, val)
                        treated = True

                if not treated:
                    for el in tmp:
                        if len(el) != 2:
                            raise AmbiguousAction("When asking for redispatching with a list, you should make a list"
                                                  "of tuple of 2 elements, the first one being the id of the"
                                                  "generator to redispatch, the second one the value of the "
                                                  "redispatching.")
                        kk, val = el
                        self.__convert_and_redispatch(kk, val)

            elif isinstance(tmp, tuple):
                if len(tmp) != 2:
                    raise AmbiguousAction("When asking for redispatching with a tuple, you should make a "
                                          "of tuple of 2 elements, the first one being the id of the"
                                          "generator to redispatch, the second one the value of the "
                                          "redispatching.")
                kk, val = tmp
                self.__convert_and_redispatch(kk, val)
            else:
                raise AmbiguousAction("Impossible to understand the redispatching action implemented.")

    def _reset_vect(self):
        """
        Need to be called when update is called !

        Returns
        -------

        """
        self._vectorized = None
        self._subs_impacted = None
        self._lines_impacted = None

    def update(self, dict_):
        """
        Update the action with a comprehensible format specified by a dictionary.

        Preferably, if a key of the argument *dict_* is not found in :attr:`Action.authorized_keys` it should throw a
        warning. This argument will be completely ignored.

        This method also reset the attributes :attr:`Action._vectorized` :attr:`Action._lines_impacted` and
        :attr:`Action._subs_impacted` to ``None`` regardless of the argument in input.

        If an action consists of "reconnecting" a powerline, and this same powerline is affected by maintenance or a
        hazard, it will be erased without any warning. "hazards" and "maintenance" have the priority. This is a
        requirement for all proper :class:`Action` subclass.

        Parameters
        ----------
        dict_: :class:`dict`
            If it's ``None`` or empty it does nothing. Otherwise, it can contain the following (optional) keys:

            - "*injection*" if the action will modify the injections (generator setpoint/load value - active or
              reactive) of the powergrid. It has optionally one of the following keys:

                    - "load_p": to set the active load values (this is a numpy array with the same size as the number of
                      load in the power _grid with Nan: don't change anything, else set the value
                    - "load_q": same as above but for the load reactive values
                    - "prod_p": same as above but for the generator active setpoint values. It has the size
                      corresponding to the number of generators in the test case.
                    - "prod_v": same as above but set the voltage setpoint of generator units.

            - "*hazards*": represents the hazards that the line might suffer (boolean vector) False: no hazard, nothing
              is done, True: a hazard, the powerline is disconnected
            - "*maintenance*": represents the maintenance operation performed on each powerline (boolean vector) False:
              no maintenance, nothing is done, True: a maintenance is scheduled, the powerline is disconnected
            - "*set_line_status*": a vector (int or float) to set the status of the powerline status (connected /
              disconnected) with the following interpretation:

                - 0: nothing is changed,
                - -1: disconnect the powerline,
                - +1: reconnect the powerline. If an action consists in "reconnecting" a powerline, and this same
                  powerline is affected by a maintenance or a hazard, it will be erased without any warning. "hazards"
                  and "maintenance" have the priority.

            - "change_line_status": a vector (bool) to change the status of the powerline. This vector should be
              interpreted as:

                - ``False``: do nothing
                - ``True``: change the status of the powerline: disconnect it if it was connected, connect it if it was
                  disconnected

            - "set_bus": (numpy int vector or dictionary) will set the buses to which the objects are connected. It
              follows a similar interpretation than the line status vector:

                - 0 -> don't change anything
                - +1 -> set to bus 1,
                - +2 -> set to bus 2, etc.
                - -1: You can use this method to disconnect an object by setting the value to -1.

            - "change_bus": (numpy bool vector or dictionary) will change the bus to which the object is connected.
              True will
              change it (eg switch it from bus 1 to bus 2 or from bus 2 to bus 1). NB this is only active if the system
              has only 2 buses per substation.

            - "redispatch" TODO

            **NB** the difference between "set_bus" and "change_bus" is the following:

              - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the
                substation to which it is connected. If it is already to bus 1 nothing will be done. If it's on another
                bus it will connect it to bus 1. It's disconnected, it will reconnect it and connect it to bus 1.
              - If "change_bus" is True, then objects will be moved from one bus to another. If the object were on bus 1
                then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object is
                disconnected then the action is ambiguous, and calling it will throw an AmbiguousAction exception.

            **NB**: if a powerline is reconnected, it should be specified on the "set_bus" action at which buses it
            should be reconnected. Otherwise, action cannot be used. Trying to apply the action to the grid will
            lead to an "AmbiguousAction" exception.

            **NB**: if for a given powerline, both switch_line_status and set_line_status is set, the action will not
            be usable.
            This will lead to an :class:`grid2op.Exception.AmbiguousAction` exception.

            **NB**: The length of vectors provided here is NOT check in this function. This method can be "chained" and
            only on the final action, when used, eg. in the Backend, is checked.

            **NB**: If a powerline is disconnected, on maintenance, or suffer an outage, the associated "set_bus" will
            be ignored.
            Disconnection has the priority on anything. This priority is given because, in case of hazard, the hazard
            has the priority over the possible actions.

        Examples
        --------
        Here are short examples on how to update an action *eg.* how to create a valid :class:`Action` object that
        be used to modify a :class:`grid2op.Backend.Backend`.

        In all the following examples, we suppose that a valid grid2op environment is created, for example with:
        .. code-block:: python

            import grid2op
            # create a simple environment
            # and make sure every type of action can be used.
            env = grid2op.make(action_class=grid2op.Action.Action)

        *Example 1*: modify the load active values to set them all to 1. You can replace "load_p" by "load_q",
        "prod_p" or "prod_v" to change the load reactive value, the generator active setpoint or the generator
        voltage magnitude setpoint.

        .. code-block:: python

            new_load = np.ones(env.action_space.n_load)
            modify_load_active_value = env.action_space({"injection": {"load_p": new_load}})
            print(modify_load_active_value)

        *Example 2*: disconnect the powerline of id 1:

        .. code-block:: python

            disconnect_powerline = env.action_space({"set_line_status": [(1, -1)]})
            print(disconnect_powerline)
            # there is a shortcut to do that:
            disconnect_powerline2 = env.disconnect_powerline(line_id=1)

        *Example 3*: force the reconnection of the powerline of id 5 by connected it to bus 1 on its origin end and
        bus 2 on its extremity end. Note that this is mandatory to specify on which bus to reconnect each
        extremity of the powerline. Otherwise it's an ambiguous action.

        .. code-block:: python

            reconnect_powerline = env.action_space({"set_line_status": [(5, 1)],
                                                    "set_bus": {"lines_or_id": [(5, 1)]},
                                                    "set_bus": {"lines_ex_id": [(5, 2)]}
                                                     })
            print(reconnect_powerline)
            # and the shorter method:
            reconnect_powerline = env.action.space.reconnect_powerline(line_id=5, bus_or=1, bus_ex=2)

        *Example 4*: change the bus to which load 4 is connected:

        .. code-block:: python

            change_load_bus = env.action_space({"set_bus": {"loads_id": [(4, 1)]} })
            print(change_load_bus)

        *Example 5*: reconfigure completely substation 5, and connect the first 3 elements to bus 1 and the last 3
        elements to bus 2

        .. code-block:: python

            sub_id = 5
            target_topology = np.ones(env.sub_info[sub_id], dtype=np.int)
            target_topology[3:] = 2
            reconfig_sub = env.action_space({"set_bus": {"substations_id": [(sub_id, target_topology)] } })
            print(reconfig_sub)

        Returns
        -------
        self: :class:`Action`
            Return the modified instance. This is handy to chain modifications if needed.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_injection(dict_)
            self._digest_redispatching(dict_)
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)

        return self

    def is_ambiguous(self):
        """
        Says if the action, as defined is ambiguous *per se* or not.

        See definition of :func:`Action.check_space_legit` for more details about *ambiguity per se*.

        Returns
        -------
        res: ``True`` if the action is ambiguous, ``False`` otherwise.

        info: ``dict`` or not
            More information about the error. If the action is not ambiguous, it values to ``None``
        """
        try:
            self._check_for_ambiguity()
            res = False
            info = None
        except AmbiguousAction as e:
            info = e
            res = True
        return res, info

    def _check_for_ambiguity(self):
        """
        This method checks if an action is ambiguous or not. If the instance is ambiguous, an
        :class:`grid2op.Exceptions.AmbiguousAction` is raised.

        An action can be ambiguous in the following context:

          - It incorrectly affects the injections:

            - :code:`self._dict_inj["load_p"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["load_q"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["prod_p"]` doesn't have the same size as the number of loads on the _grid.
            - :code:`self._dict_inj["prod_v"]` doesn't have the same size as the number of loads on the _grid.

          - It affects the powerline in an incorrect manner:

            - :code:`self._switch_line_status` has not the same size as the number of powerlines
            - :code:`self._set_line_status` has not the same size as the number of powerlines
            - somes lines are reconnected (:code:`self._switch_line_status[i] == True` for some powerline *i* but it is
              not specified on which bus to connect it ( the element corresponding to powerline *i* in
              :code:`self._set_topo_vect` is set to 0)
            - the status of some powerline is both *changed* (:code:`self._switch_line_status[i] == True` for some *i*)
              and *set* (:code:`self._set_line_status[i]` for the same *i* is not 0)

          - It has an ambiguous behavior concerning the topology of some substations

            - the state of some bus for some element is both *changed* (:code:`self._change_bus_vect[i] = True` for
              some *i*) and *set* (:code:`self._set_topo_vect[i]` for the same *i* is not 0)
            - :code:`self._set_topo_vect` has not the same dimension as the number of elements on the powergrid
            - :code:`self._change_bus_vect` has not the same dimension as the number of elements on the powergrid

          - For redispatching, Ambiguous actions can come from:

            - Some redispatching action is active, yet
              :attr:`grid2op.Space.GridObjects.redispatching_unit_commitment_availble` is set to ``False``
            - the length of the redispatching vector :attr:`Action._redispatching` is not compatible with the number
              of generators.
            - some redispatching are above the maximum ramp up :attr:`grid2op.Space.GridObjects.gen_max_ramp_up`
            - some redispatching are below the maximum ramp down :attr:`grid2op.Space.GridObjects.gen_max_ramp_down`
            - the redispatching action affect non dispatchable generators
            - the redispatching and the production setpoint, if added, are above pmax for at least a generator
            - the redispatching and the production setpoint, if added, are below pmin for at least a generator

        In case of need to overload this method, it is advise to still call this one from the base :class:`Action`
        with ":code:`super()._check_for_ambiguity()`" or ":code:`Action._check_for_ambiguity(self)`".

        Raises
        -------
        :class:`grid2op.Exceptions.AmbiguousAction`
            Or any of its more precise subclasses, depending on which assumption is not met.


        """
        if np.any(self._set_line_status[self._switch_line_status] != 0):
            raise InvalidLineStatus("You asked to change the status (connected / disconnected) of a powerline by"
                                    " using the keyword \"change_status\" and set this same line state in "
                                    "\"set_status\" "
                                    "(or \"hazard\" or \"maintenance\"). This ambiguous behaviour is not supported")
        # check size
        if "load_p" in self._dict_inj:
            if len(self._dict_inj["load_p"]) != self.n_load:
                raise InvalidNumberOfLoads("This action acts on {} loads while there are {} "
                                           "in the _grid".format(len(self._dict_inj["load_p"]), self.n_load))
        if "load_q" in self._dict_inj:
            if len(self._dict_inj["load_q"]) != self.n_load:
                raise InvalidNumberOfLoads("This action acts on {} loads while there are {} in "
                                           "the _grid".format(len(self._dict_inj["load_q"]), self.n_load))
        if "prod_p" in self._dict_inj:
            if len(self._dict_inj["prod_p"]) != self.n_gen:
                raise InvalidNumberOfGenerators("This action acts on {} generators while there are {} in "
                                                "the _grid".format(len(self._dict_inj["prod_p"]), self.n_gen))
        if "prod_v" in self._dict_inj:
            if len(self._dict_inj["prod_v"]) != self.n_gen:
                raise InvalidNumberOfGenerators("This action acts on {} generators while there are {} in "
                                                "the _grid".format(len(self._dict_inj["prod_v"]), self.n_gen))

        if len(self._switch_line_status) != self.n_line:
                raise InvalidNumberOfLines("This action acts on {} lines while there are {} in "
                                           "the _grid".format(len(self._switch_line_status), self.n_line))

        if len(self._set_topo_vect) != self.dim_topo:
                raise InvalidNumberOfObjectEnds("This action acts on {} ends of object while there are {} "
                                                "in the _grid".format(len(self._set_topo_vect), self.dim_topo))
        if len(self._change_bus_vect) != self.dim_topo:
                raise InvalidNumberOfObjectEnds("This action acts on {} ends of object while there are {} "
                                                "in the _grid".format(len(self._change_bus_vect), self.dim_topo))

        if len(self._redispatch) != self.n_gen:
            raise InvalidNumberOfGenerators("This action acts on {} generators (redispatching= while "
                                            "there are {} in the grid".format(len(self._redispatch), self.n_gen))

        # redispatching specific check
        if np.any(self._redispatch != 0.):
            if not self.redispatching_unit_commitment_availble:
                raise UnitCommitorRedispachingNotAvailable("Impossible to use a redispatching action in this "
                                                           "environment. Please set up the proper costs for generator")

            if np.any(self._redispatch[~self.gen_redispatchable] != 0.):
                raise InvalidRedispatching("Trying to apply a redispatching action on a non redispatchable generator")

            if self._single_act:
                # TODO check that when action is made (and check also the buses id, don't put 3 for example...)
                if np.any(self._redispatch > self.gen_max_ramp_up):
                   raise InvalidRedispatching("Some redispatching amount are above the maximum ramp up")
                if np.any(-self._redispatch > self.gen_max_ramp_down):
                   raise InvalidRedispatching("Some redispatching amount are bellow the maximum ramp down")

                if "prod_p" in self._dict_inj:
                    new_p = self._dict_inj["prod_p"]
                    tmp_p = new_p + self._redispatch
                    indx_ok = np.isfinite(new_p)
                    if np.any(tmp_p[indx_ok] > self.gen_pmax[indx_ok]):
                        raise InvalidRedispatching("Some redispatching amount, cumulated with the production setpoint, "
                                                   "are above pmax for some generator.")
                    if np.any(tmp_p[indx_ok] < self.gen_pmin[indx_ok]):
                        raise InvalidRedispatching("Some redispatching amount, cumulated with the production setpoint, "
                                                   "are below pmin for some generator.")

        # topological action
        if np.any(self._set_topo_vect[self._change_bus_vect] != 0):
            raise InvalidBusStatus("You asked to change the bus of an object with"
                                   " using the keyword \"change_bus\" and set this same object state in \"set_bus\""
                                   ". This ambiguous behaviour is not supported")
        if np.any(self._set_topo_vect < -1):
            raise InvalidBusStatus("Invalid set_bus. Buses should be either -1 (disconnect), 0 (change nothing),"
                                   "1 (assign this object to bus one) or 2 (assign this object to bus"
                                   "2). A negative number has been found.")
        if np.any(self._set_topo_vect > 2):
            raise InvalidBusStatus("Invalid set_bus. Buses should be either -1 (disconnect), 0 (change nothing),"
                                   "1 (assign this object to bus one) or 2 (assign this object to bus"
                                   "2). A number higher than 2 has been found: substations with more than 2 busbars"
                                   "are not supported by grid2op.")

        for q_id, status in enumerate(self._set_line_status):
            if status == 1:
                # i reconnect a powerline, i need to check that it's connected on both ends
                if self._set_topo_vect[self.line_or_pos_topo_vect[q_id]] == 0 or \
                        self._set_topo_vect[self.line_ex_pos_topo_vect[q_id]] == 0:

                    raise InvalidLineStatus("You ask to reconnect powerline {} yet didn't tell on"
                                            " which bus.".format(q_id))

        # if i disconnected of a line, but i modify also the bus where it's connected
        idx = self._set_line_status == -1
        id_disc = np.where(idx)[0]
        if np.any(self._set_topo_vect[self.line_or_pos_topo_vect[id_disc]] > 0) or \
                np.any(self._set_topo_vect[self.line_ex_pos_topo_vect[id_disc]] > 0):
                    raise InvalidLineStatus("You ask to disconnect a powerline but also to connect it "
                                            "to a certain bus.")
        if np.any(self._change_bus_vect[self.line_or_pos_topo_vect[id_disc]] > 0) or \
                np.any(self._change_bus_vect[self.line_ex_pos_topo_vect[id_disc]] > 0):
            raise InvalidLineStatus("You ask to disconnect a powerline but also to change its bus.")

        if np.any(self._change_bus_vect[self.line_or_pos_topo_vect[self._set_line_status == 1]]):
            raise InvalidLineStatus("You ask to connect an origin powerline but also to *change* the bus  to which it "
                                    "is connected. This is ambiguous. You must *set* this bus instead.")
        if np.any(self._change_bus_vect[self.line_ex_pos_topo_vect[self._set_line_status == 1]]):
            raise InvalidLineStatus("You ask to connect an extremity powerline but also to *change* the bus  to which "
                                    "it is connected. This is ambiguous. You must *set* this bus instead.")

    def sample(self, space_prng):
        """
        This method is used to sample action.

        A generic sampling of action can be really tedious. Uniform sampling is almost impossible.
        The actual implementation gives absolutely no warranty toward any of these concerns.

        It is not implemented yet.
        TODO

        By calling :func:`Action.sample`, the action is :func:`Action.reset` to a "do nothing" state.

        Parameters
        ----------
        space_prng

        Returns
        -------
        self: :class:`Action`
            The action sampled among the action space.
        """
        self.reset()
        # TODO code the sampling now
        return self

    def _ignore_topo_action_if_disconnection(self, sel_):
        # force ignore of any topological actions
        self._set_topo_vect[np.array(self.line_or_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self.line_or_pos_topo_vect[sel_])] = False
        self._set_topo_vect[np.array(self.line_ex_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self.line_ex_pos_topo_vect[sel_])] = False

    def _obj_caract_from_topo_id(self, id_):
        obj_id = None
        objt_type = None
        array_subid = None
        for l_id, id_in_topo in enumerate(self.load_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                objt_type = "load"
                array_subid = self.load_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.gen_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "generator"
                    array_subid = self.gen_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.line_or_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "line (origin)"
                    array_subid = self.line_or_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self.line_ex_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "line (extremity)"
                    array_subid = self.line_ex_to_subid
        substation_id = array_subid[obj_id]
        return obj_id, objt_type, substation_id

    def __str__(self):
        """
        This utility allows printing in a human-readable format what objects will be impacted by the action.

        Returns
        -------
        str: :class:`str`
            The string representation of an :class:`Action` in a human-readable format.

        """
        res = ["This action will:"]
        impact = self.impact_on_objects()

        # injections
        injection_impact = impact['injection']
        if injection_impact['changed']:
            for change in injection_impact['impacted']:
                res.append("\t - set {} to {}".format(change['set'], change['to']))
        else:
            res.append("\t - NOT change anything to the injections")
        
        if np.any(self._redispatch != 0.):
            res.append("\t - perform the following redispatching action: {}".format(self._redispatch))
        else:
            res.append("\t - NOT perform any redispatching action")

        # force line status
        force_line_impact = impact['force_line']
        if force_line_impact['changed']:
            reconnections = force_line_impact['reconnections']
            if reconnections['count'] > 0:
                res.append("\t - force reconnection of {} powerlines ({})"
                           .format(reconnections['count'], reconnections['powerlines']))

            disconnections = force_line_impact['disconnections']
            if disconnections['count'] > 0:
                res.append("\t - force disconnection of {} powerlines ({})"
                           .format(disconnections['count'], disconnections['powerlines']))
        else:
            res.append("\t - NOT force any line status")

        # swtich line status
        swith_line_impact = impact['switch_line']
        if swith_line_impact['changed']:
            res.append("\t - switch status of {} powerlines ({})"
                       .format(swith_line_impact['count'], swith_line_impact['powerlines']))
        else:
            res.append("\t - NOT switch any line status")

        # topology
        bus_switch_impact = impact['topology']['bus_switch']
        if len(bus_switch_impact) > 0:
            res.append("\t - Change the bus of the following element:")
            for switch in bus_switch_impact:
                res.append("\t \t - switch bus of {} {} [on substation {}]"
                           .format(switch['object_type'], switch['object_id'],
                                   switch['substation']))
        else:
            res.append("\t - NOT switch anything in the topology")

        assigned_bus_impact = impact['topology']['assigned_bus']
        disconnect_bus_impact = impact['topology']['disconnect_bus']
        if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
            res.append("\t - Set the bus of the following element:")
            for assigned in assigned_bus_impact:
                res.append("\t \t - assign bus {} to {} {} [on substation {}]"
                           .format(assigned['bus'], assigned['object_type'], assigned['object_id'],
                                   assigned['substation']))

            for disconnected in disconnect_bus_impact:
                res.append("\t - disconnect {} {} [on substation {}]"
                           .format(disconnected['object_type'], disconnected['object_id'],
                                   disconnected['substation']))

        else:
            res.append("\t - NOT force any particular bus configuration")

        return "\n".join(res)

    def impact_on_objects(self):
        """
        This will return a dictionary which contains details on objects that will be impacted by the action.

        Returns
        -------
        dict: :class:`dict`
            The dictionary representation of an action impact on objects

        """
        # TODO: include redispatching 
         
        # handles actions on injections
        has_impact = False

        inject_detail = {
            'changed': False,
            'count': 0,
            'impacted': []
        }
        for k in ["load_p", "prod_p", "load_q", "prod_v"]:
            if k in self._dict_inj:
                inject_detail['changed'] = True
                has_impact = True
                inject_detail['count'] += 1
                inject_detail['impacted'].append({
                    'set': k,
                    'to': self._dict_inj[k]
                })

        # handles actions on force line status
        force_line_status = {
            'changed': False,
            'reconnections': {'count': 0, 'powerlines': []},
            'disconnections': {'count': 0, 'powerlines': []}
        }
        if np.any(self._set_line_status == 1):
            force_line_status['changed'] = True
            has_impact = True
            force_line_status['reconnections']['count'] = np.sum(self._set_line_status == 1)
            force_line_status['reconnections']['powerlines'] = np.where(self._set_line_status == 1)[0]

        if np.any(self._set_line_status == -1):
            force_line_status['changed'] = True
            has_impact = True
            force_line_status['disconnections']['count'] = np.sum(self._set_line_status == -1)
            force_line_status['disconnections']['powerlines'] = np.where(self._set_line_status == -1)[0]

        # handles action on swtich line status
        switch_line_status = {
            'changed': False,
            'count': 0,
            'powerlines': []
        }
        if np.sum(self._switch_line_status):
            switch_line_status['changed'] = True
            has_impact = True
            switch_line_status['count'] = np.sum(self._switch_line_status)
            switch_line_status['powerlines'] = np.where(self._switch_line_status)[0]

        topology = {
            'changed': False,
            'bus_switch': [],
            'assigned_bus': [],
            'disconnect_bus': []
        }
        # handles topology
        if np.any(self._change_bus_vect):
            for id_, k in enumerate(self._change_bus_vect):
                if k:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    topology['bus_switch'].append({
                        'bus': k,
                        'object_type': objt_type,
                        'object_id': obj_id,
                        'substation': substation_id
                    })
            topology['changed'] = True
            has_impact = True

        if np.any(self._set_topo_vect != 0):
            for id_, k in enumerate(self._set_topo_vect):
                if k > 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    topology['assigned_bus'].append({
                        'bus': k,
                        'object_type': objt_type,
                        'object_id': obj_id,
                        'substation': substation_id
                    })

                if k < 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    topology['disconnect_bus'].append({
                        'bus': k,
                        'object_type': objt_type,
                        'object_id': obj_id,
                        'substation': substation_id
                    })
            topology['changed'] = True
            has_impact = True

        return {
            'has_impact': has_impact,
            'injection': inject_detail,
            'force_line': force_line_status,
            'switch_line': switch_line_status,
            'topology': topology
        }

    def as_dict(self):
        """
        Represent an action "as a" dictionary. This dictionary is useful to further inspect on which elements
        the actions had an impact. It is not recommended to use it as a way to serialize actions. The "do nothing" action
        should always be represented by an empty dictionary.

        The following keys (all optional) are present in the results:

          * `load_p`: if the action modifies the active loads.
          * `load_q`: if the action modifies the reactive loads.
          * `prod_p`: if the action modifies the active productions of generators.
          * `prod_v`: if the action modifies the voltage setpoint of generators.
          * `set_line_status` if the action tries to **set** the status of some powerlines. If present, this is a
            a dictionary with keys:

              * `nb_connected`: number of powerlines that are reconnected
              * `nb_disconnected`: number of powerlines that are disconnected
              * `connected_id`: the id of the powerlines reconnected
              * `disconnected_id`: the ids of the powerlines disconnected

          * `change_line_status`: if the action tries to **change** the status of some powerlines. If present, this
            is a dictionary with keys:

              * `nb_changed`: number of powerlines having their status changed
              * `changed_id`: the ids of the powerlines that are changed

          * `change_bus_vect`: if the action tries to **change** the topology of some substations. If present, this
            is a dictionary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: ids of the substations impacted by the action
              * `change_bus_vect`: details the objects that are modified. It is itself a dictionary that represents for
                each impacted substations (keys) the modification of the objects connected to it.

          * `set_bus_vect`: if the action tries to **set** the topology of some substations. If present, this is a
            dictionary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: the ids of the substations impacted by the action
              * `set_bus_vect`: details the objects that are modified. It is also a dictionary that represents for
                each impacted substations (keys) how the elements connected to it are impacted (their "new" bus)

          * `hazards` if the action is composed of some hazards. In this case, it's simply the index of the powerlines
            that are disconnected because of them.
          * `nb_hazards` the number of hazards the "action" implemented (eg number of powerlines disconnected because of
            hazards.
          * `maintenance` if the action is composed of some maintenance. In this case, it's simply the index of the
            powerlines that are affected by maintenance operation at this time step.
            that are disconnected because of them.
          * `nb_maintenance` the number of maintenance the "action" implemented eg the number of powerlines
            disconnected because of maintenance operations.
          * `redispatch` the redispatching action (if any). It gives, for each generator (all generator, not just the
            dispatchable one) the amount of power redispatched in this action.

        Returns
        -------
        res: ``dict``
            The action represented as a dictionary. See above for a description of it.


        """
        res = {}

        # saving the injections
        for k in ["load_p", "prod_p", "load_q", "prod_v"]:
            if k in self._dict_inj:
                res[k] = self._dict_inj[k]

        # handles actions on force line status
        if np.any(self._set_line_status != 0):
            res["set_line_status"] = {}
            res["set_line_status"]["nb_connected"] = np.sum(self._set_line_status == 1)
            res["set_line_status"]["nb_disconnected"] = np.sum(self._set_line_status == -1)
            res["set_line_status"]["connected_id"] = np.where(self._set_line_status == 1)[0]
            res["set_line_status"]["disconnected_id"] = np.where(self._set_line_status == -1)[0]

        # handles action on swtich line status
        if np.sum(self._switch_line_status):
            res["change_line_status"] = {}
            res["change_line_status"]["nb_changed"] = np.sum(self._switch_line_status)
            res["change_line_status"]["changed_id"] = np.where(self._switch_line_status)[0]

        # handles topology change
        if np.any(self._change_bus_vect):
            res["change_bus_vect"] = {}
            res["change_bus_vect"]["nb_modif_objects"] = np.sum(self._change_bus_vect)
            all_subs = set()
            for id_, k in enumerate(self._change_bus_vect):
                if k:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    sub_id = "{}".format(substation_id)
                    if not sub_id in res["change_bus_vect"]:
                        res["change_bus_vect"][sub_id] = {}
                    res["change_bus_vect"][sub_id]["{}".format(obj_id)] = {"type": objt_type}
                    all_subs.add(sub_id)

            res["change_bus_vect"]["nb_modif_subs"] = len(all_subs)
            res["change_bus_vect"]["modif_subs_id"] = sorted(all_subs)

        # handles topology set
        if np.any(self._set_topo_vect):
            res["set_bus_vect"] = {}
            res["set_bus_vect"]["nb_modif_objects"] = np.sum(self._set_topo_vect)
            all_subs = set()
            for id_, k in enumerate(self._set_topo_vect):
                if k != 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    sub_id = "{}".format(substation_id)
                    if not sub_id in res["set_bus_vect"]:
                        res["set_bus_vect"][sub_id] = {}
                    res["set_bus_vect"][sub_id]["{}".format(obj_id)] = {"type": objt_type, "new_bus": k}
                    all_subs.add(sub_id)

            res["set_bus_vect"]["nb_modif_subs"] = len(all_subs)
            res["set_bus_vect"]["modif_subs_id"] = sorted(all_subs)

        if np.any(self._hazards):
            res["hazards"] = np.where(self._hazards)[0]
            res["nb_hazards"] = np.sum(self._hazards)

        if np.any(self._maintenance):
            res["maintenance"] = np.where(self._maintenance)[0]
            res["nb_maintenance"] = np.sum(self._maintenance)

        if np.any(self._redispatch != 0.):
            res["redispatch"] = self._redispatch

        return res

    def effect_on(self, _sentinel=None, load_id=None, gen_id=None, line_id=None, substation_id=None):
        """
        Return the effect of this action on a unique given load, generator unit, powerline or substation.
        Only one of load, gen, line or substation should be filled.

        The query of these objects can only be done by id here (ie by giving the integer of the object in the backed).
        The :class:`HelperAction` has some utilities to access them by name too.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, **do not use**.

        load_id: ``int``
            The ID of the load we want to inspect

        gen_id: ``int``
            The ID of the generator we want to inspect

        line_id: ``int``
            The ID of the powerline we want to inspect

        substation_id: ``int``
            The ID of the substation we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionary with keys and value depending on which object needs to be inspected:

            - if a load is inspected, then the keys are:

                - "new_p" the new load active value (or NaN if it doesn't change),
                - "new_q" the new load reactive value (or Nan if nothing has changed from this point of view)
                - "set_bus" the new bus where the load will be moved (int: id of the bus, 0 no change, -1 disconnected)
                - "change_bus" whether or not this load will be moved from one bus to another (for example is an action
                  asked it to go from bus 1 to bus 2)

            - if a generator is inspected, then the keys are:

                - "new_p" the new generator active setpoint value (or NaN if it doesn't change),
                - "new_v" the new generator voltage setpoint value (or Nan if nothing has changed from this point of
                  view)
                - "set_bus" the new bus where the load will be moved (int: id of the bus, 0 no change, -1 disconnected)
                - "change_bus" whether or not this load will be moved from one bus to another (for example is an action
                  asked it to go from bus 1 to bus 2)
                - "redispatch" the amount of power redispatched for this generator.

            - if a powerline is inspected then the keys are:

                - "change_bus_or": whether or not the origin end will be moved from one bus to another
                - "change_bus_ex": whether or not the extremity end will be moved from one bus to another
                - "set_bus_or": the new bus where the origin will be moved
                - "set_bus_ex": the new bus where the extremity will be moved
                - "set_line_status": the new status of the power line
                - "change_line_status": whether or not to switch the status of the powerline

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "change_bus"
                - "set_bus"

        NB the difference between "set_bus" and "change_bus" is the following:

            - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the substation
              to which it is connected. If it is already to bus 1 nothing will be done. If it's on another bus it will
              connect it to bus 1. It's disconnected, it will reconnect it and connect it to bus 1.
            - If "change_bus" is True, then the object will be moved from one bus to another. If the object were on bus 1
              then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
              disconnected, then it will be connected to the affected bus.

        Raises
        ------
        :class:`grid2op.Exception.Grid2OpException`
            If _sentinel is modified, or if none of the arguments are set or alternatively if 2 or more of the
            parameters are being set.

        """
        if _sentinel is not None:
            raise Grid2OpException("action.effect_on should only be called with named argument.")

        if load_id is None and gen_id is None and line_id is None and substation_id is None:
            raise Grid2OpException("You ask the effect of an action on something, wihtout provided anything")

        if load_id is not None:
            if gen_id is not None or line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inpsect the effect of an action on one single element")
            res = {"new_p": np.NaN, "new_q": np.NaN, "change_bus": False, "set_bus": 0}
            if "load_p" in self._dict_inj:
                res["new_p"] = self._dict_inj["load_p"][load_id]
            if "load_q" in self._dict_inj:
                res["new_q"] = self._dict_inj["load_q"][load_id]
            my_id = self.load_pos_topo_vect[load_id]
            res["change_bus"] = self._change_bus_vect[my_id]
            res["set_bus"] = self._set_topo_vect[my_id]

        elif gen_id is not None:
            if line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inpsect the effect of an action on one single element")
            res = {"new_p": np.NaN, "new_v": np.NaN, "set_bus": 0., "change_bus": False}
            if "prod_p" in self._dict_inj:
                res["new_p"] = self._dict_inj["prod_p"][gen_id]
            if "prod_v" in self._dict_inj:
                res["new_v"] = self._dict_inj["prod_v"][gen_id]
            my_id = self.gen_pos_topo_vect[gen_id]
            res["change_bus"] = self._change_bus_vect[my_id]
            res["set_bus"] = self._set_topo_vect[my_id]
            res["redispatch"] = self._redispatch[gen_id]

        elif line_id is not None:
            if substation_id is not None:
                raise Grid2OpException("You can only the inpsect the effect of an action on one single element")
            res = {}
            # origin topology
            my_id = self.line_or_pos_topo_vect[line_id]
            res["change_bus_or"] = self._change_bus_vect[my_id]
            res["set_bus_or"] = self._set_topo_vect[my_id]
            # extremity topology
            my_id = self.line_ex_pos_topo_vect[line_id]
            res["change_bus_ex"] = self._change_bus_vect[my_id]
            res["set_bus_ex"] = self._set_topo_vect[my_id]
            # status
            res["set_line_status"] = self._set_line_status[line_id]
            res["change_line_status"] = self._switch_line_status[line_id]
        else:
            res = {}
            beg_ = int(np.sum(self.sub_info[:substation_id]))
            end_ = int(beg_ + self.sub_info[substation_id])
            res["change_bus"] = self._change_bus_vect[beg_:end_]
            res["set_bus"] = self._set_topo_vect[beg_:end_]

        return res


class TopoAndRedispAction(Action):
    def __init__(self, gridobj):
        Action.__init__(self, gridobj)
        self.authorized_keys = set([k for k in self.authorized_keys if k != "injection"])

        self.attr_list_vect = ["_set_line_status", "_switch_line_status",
                               "_set_topo_vect", "_change_bus_vect",
                               "_redispatch"]

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionary.

        Returns
        -------
        dict_injection: ``dict``
            This dictionary is always empty

        set_line_status: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`Action._switch_line_status`

        set_topo_vect: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`Action._set_topo_vect`

        change_bus_vect: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`Action._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            Thie array is :attr:`Action._redispatch`

        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect,\
               self._redispatch

    def update(self, dict_):
        """
        As its original implementation, this method allows modifying the way a dictionary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection" part is irrelevant for this subclass.

        Returns
        -------
        self: :class:`TopologyAction`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """

        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            # self._digest_injection(dict_)  # only difference compared to the base class implementation.
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_change_status(dict_)
            self._digest_redispatching(dict_)
        return self


class TopologyAction(Action):
    """
    This class is model only topological actions.
    It will throw an ":class:`grid2op.Exception.AmbiguousAction`" error it someone attempt to change injections
    in any ways.

    It has the same attributes as its base class :class:`Action`.

    It is also here to show an example on how to implement a valid class deriving from :class:`Action`.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.

        """
        Action.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys if k != "injection" and k != "redispatch"])

        self.attr_list_vect = ["_set_line_status", "_switch_line_status",
                               "_set_topo_vect", "_change_bus_vect"]

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionary.

        Returns
        -------
        dict_injection: ``dict``
            This dictionary is always empty

        set_line_status: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`Action._switch_line_status`

        set_topo_vect: :class:`numpy.ndarray`, dtype:int
            This array is :attr:`Action._set_topo_vect`

        change_bus_vect: :class:`numpy.ndarray`, dtype:bool
            This array is :attr:`Action._change_bus_vect`

        redispatch: :class:`numpy.ndarray`, dtype:float
            Thie array is :attr:`Action._redispatch`

        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect,\
               self._redispatch

    def update(self, dict_):
        """
        As its original implementation, this method allows modifying the way a dictionary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection" part is irrelevant for this subclass.

        Returns
        -------
        self: :class:`TopologyAction`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            # self._digest_injection(dict_)  # only difference compared to the base class implementation.
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_change_status(dict_)
        return self

    def sample(self, space_prng):
        """
        Sample a Topology action.

        This method is not implemented at the moment. TODO

        Parameters
        ----------
        space_prng

        Returns
        -------
        res: :class:`TopologyAction`
            The current action (useful to chain some calls to methods)
        """
        self.reset()
        return self


class VoltageOnlyAction(Action):
    """
    This class is here to serve as a base class for the controler of the voltages (if any). It allows to perform
    only modification of the generator voltage set point.

    Only action of type "injection" are supported, and only setting "prod_v" keyword.
    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.

        """
        Action.__init__(self, gridobj)
        self.authorized_keys = {"injection"}
        self.attr_list_vect = ["prod_v"]

    def _check_dict(self):
        """
        Check that nothing, beside prod_v has been updated with this action.

        Returns
        -------

        """
        if self._dict_inj:
            for el in self._dict_inj:
                if el not in self.attr_list_vect:
                    raise AmbiguousAction("Impossible to modify something different than \"prod_v\" using "
                                          "\"VoltageOnlyAction\" action.")

    def update(self, dict_):
        """
        As its original implementation, this method allows modifying the way a dictionary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection", "change bus", "set bus", or "change line status" are irrelevant for this subclass.

        Returns
        -------
        self: :class:`PowerLineSet`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_injection(dict_)
            self._check_dict()
        return self


class PowerLineSet(Action):
    """
    This class is here to model only a subpart of Topological actions, the one consisting of topological switching.
    It will throw an "AmbiguousAction" error if someone attempts to change injections in any way.

    It has the same attributes as its base class :class:`Action`.

    It is also here to show an example of how to implement a valid class deriving from :class:`Action`.

    **NB** This class doesn't allow to connect an object to other buses than their original bus. In this case,
    reconnecting a powerline cannot be considered "ambiguous": all powerlines are reconnected on bus 1 on both
    of their substations.

    """

    def __init__(self, gridobj):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.

        """
        Action.__init__(self, gridobj)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys
                                    if k != "injection" and k != "set_bus" and
                                    k != "change_bus" and k != "change_line_status" and
                                    k != "redispatch"])

        self.attr_list_vect = ["_set_line_status"]

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionary is always empty

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._switch_line_status`, it is never modified

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_topo_vect`, it is never modified

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._change_bus_vect`, it is never modified

        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect, \
               self._redispatch

    def update(self, dict_):
        """
        As its original implementation, this method allows modifying the way a dictionary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection", "change bus", "set bus", or "change line status" are irrelevant for this subclass.

        Returns
        -------
        self: :class:`PowerLineSet`
            Return object itself thus allowing multiple calls to "update" to be chained.

        """
        self._reset_vect()

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_set_status(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
        self.disambiguate_reconnection()

        return self

    def check_space_legit(self):
        self.disambiguate_reconnection()
        self._check_for_ambiguity()

    def disambiguate_reconnection(self):
        """
        As this class doesn't allow to perform any topology change, when a powerline is reconnected, it's necessarily
        on the first bus of the substation.

        So it's not ambiguous in this case. We have to implement this logic here, and that is what is done in this
        function.

        """
        sel_ = self._set_line_status == 1
        if np.any(sel_):
            self._set_topo_vect[self.line_ex_pos_topo_vect[sel_]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[sel_]] = 1

    def sample(self, space_prng):
        """
        Sample a PowerlineSwitch Action.

        By default, this sampling will act on one random powerline, and it will either
        disconnect it or reconnect it each with equal probability.

        Parameters
        ----------
        space_prng: ``numpy.random.RandomState``
            The pseudo random number generator of the Action space used to sample actions.

        Returns
        -------
        res: :class:`PowerLineSwitch`
            The sampled action
            
        """
        self.reset()
        # TODO here use the prng state from the ActionSpace !!!!
        i = space_prng.randint(0, self.size())  # the powerline on which we can act
        val = 2*np.random.randint(0, 2) - 1  # the action: +1 reconnect it, -1 disconnect it
        self._set_line_status[i] = val
        if val == 1:
            self._set_topo_vect[self.line_ex_pos_topo_vect[i]] = 1
            self._set_topo_vect[self.line_or_pos_topo_vect[i]] = 1
        return self


class SerializableActionSpace(SerializableSpace):
    """
    This class allows serializing/ deserializing the action space.

    It should not be used inside an :attr:`grid2op.Environment.Environment` , as some functions of the action might not
    be compatible with the serialization, especially the checking of whether or not an action is legal or not.

    Attributes
    ----------

    actionClass: ``type``
        Type used to build the :attr:`SerializableActionSpace.template_act`

    _template_act: :class:`Action`
        An instance of the "*actionClass*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`Action.size`) or to sample a new Action (see :func:`grid2op.Action.Action.sample`)

    """
    def __init__(self, gridobj, actionClass=Action):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the underlying powergrid.

        actionClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`. It should derived from
            :class:`Action`.

        """
        SerializableSpace.__init__(self, gridobj=gridobj, subtype=actionClass)

        self.actionClass = self.subtype
        self._template_act = self._template_obj

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionary (for example in the case of JSON saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an Action Space (aka SerializableActionSpace) as a dictionary.

        Returns
        -------
        res: :class:``SerializableActionSpace``
            An instance of an action space matching the dictionary.


        """
        tmp = SerializableSpace.from_dict(dict_)
        res = SerializableActionSpace(gridobj=tmp, actionClass=tmp.subtype)
        return res

    def sample(self):
        """
        A utility used to sample :class:`Action`.

        This method is under development, use with care (actions are not sampled on the full action space, and are
        not uniform in general).

        Returns
        -------
        res: :class:`Action`
            A random action sampled from the :attr:`HelperAction.actionClass`

        """
        res = self.actionClass(gridobj=self)  # only the GridObjects part of "self" is actually used
        res.sample()
        return res

    def disconnect_powerline(self, line_id, previous_action=None):
        """
        Utilities to disconnect a powerline more easily.

        Parameters
        ----------
        line_id: ``int``
            The powerline to be disconnected.

        previous_action

        Returns
        -------

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `HelperAction` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction("You asked to disconnect powerline of id {} but this id does not exist. The "
                                  "grid counts only {} powerline".format(line_id, self.n_line))
        res.update({"set_line_status": [(line_id, -1)]})
        return res

    def reconnect_powerline(self, line_id, bus_or, bus_ex, previous_action=None):
        """
        Utilities to reconnect a powerline more easily.

        Note that in case "bus_or" or "bus_ex" are not the current bus to which the powerline is connected, they
        will be affected by this action.

        Parameters
        ----------
        line_id: ``int``
            The powerline to be disconnected.

        bus_or: ``int``
            On which bus to reconnect the powerline at its origin end

        bus_ex: ``int``
            On which bus to reconnect the powerline at its extremity end
        previous_action

        Returns
        -------

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `HelperAction` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action
        if line_id > self.n_line:
            raise AmbiguousAction("You asked to disconnect powerline of id {} but this id does not exist. The "
                                  "grid counts only {} powerline".format(line_id, self.n_line))
        res.update({"set_line_status": [(line_id, 1)],
                    "set_bus": {"lines_or_id": [(line_id, bus_or)],
                                "lines_ex_id": [(line_id, bus_ex)]}})
        return res

    def change_bus(self, name_element, extremity=None, substation=None, type_element=None, previous_action=None):
        """
        Utilities to change the bus of a single element if you give its name. **NB** Changing a bus has the effect to
        assign the object to bus 1 if it was before that connected to bus 2, and to assign it to bus 2 if it was
        connected to bus 1. It should not be mixed up with :func:`HelperAction.set_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (in place) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus
        extremity: ``str``
            "or" or "ex" for origin or extremity, ignored if an element is not a powerline.
        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise, the method will search for it.
        type_element: ``int``, optional
            Type of the element to look for. It is here to speed up the computation. One of "line", "gen" or "load"
        previous_action: :class:`Action`, optional
            The (optional) action to update. It should be of the same type as :attr:`HelperAction.actionClass`

        Returns
        -------
        res: :class:`Action`
            The action with the modification implemented

        Raises
        ------
        :class:`grid2op.Exception.AmbiguousAction`
            If *previous_action* has not the same type as :attr:`HelperAction.actionClass`.

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `HelperAction` is of type \"{}\" "
                                      "which is not the type of action handled by this helper "
                                      "(\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation,
                                                                        type_element, res)
        dict_["change_bus"][to_sub_pos[my_id]] = True
        res.update({"change_bus": {"substations_id": [(my_sub_id, dict_["change_bus"])]}})
        # res.update(dict_)
        return res

    def _extract_database_powerline(self, extremity):
        if extremity[:2] == "or":
            to_subid = self.line_or_to_subid
            to_sub_pos = self.line_or_to_sub_pos
            to_name = self.name_line
        elif extremity[:2] == "ex":
            to_subid = self.line_ex_to_subid
            to_sub_pos = self.line_ex_to_sub_pos
            to_name = self.name_line
        elif extremity is None:
            raise Grid2OpException("It is mandatory to know on which ends you want to change the bus of the powerline")
        else:
            raise Grid2OpException("unknown extremity specifier \"{}\". Extremity should be \"or\" or \"ex\""
                                   "".format(extremity))
        return to_subid, to_sub_pos, to_name

    def _extract_dict_action(self, name_element, extremity=None, substation=None, type_element=None, action=None):
        to_subid = None
        to_sub_pos = None
        to_name = None

        if type_element == "line":
            to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
        elif type_element[:3] == "gen" or type_element[:4] == "prod":
            to_subid = self.gen_to_subid
            to_sub_pos = self.gen_to_sub_pos
            to_name = self.name_gen
        elif type_element == "load":
            to_subid = self.load_to_subid
            to_sub_pos = self.load_to_sub_pos
            to_name = self.name_load
        elif type_element is None:
            # i have to look through all the objects to find it
            if name_element in self.name_load:
                to_subid = self.load_to_subid
                to_sub_pos = self.load_to_sub_pos
                to_name = self.name_load
            elif name_element in self.name_gen:
                to_subid = self.gen_to_subid
                to_sub_pos = self.gen_to_sub_pos
                to_name = self.name_gen
            elif name_element in self.name_line:
                to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
            else:
                AmbiguousAction(
                    "Element \"{}\" not found in the powergrid".format(
                        name_element))
        else:
            raise AmbiguousAction("unknown type_element specifier \"{}\". type_element should be \"line\" or \"load\" "
                                  "or \"gen\"".format(extremity))

        my_id = None
        for i, nm in enumerate(to_name):
            if nm == name_element:
                my_id = i
                break
        if my_id is None:
            raise AmbiguousAction("Element \"{}\" not found in the powergrid".format(name_element))
        my_sub_id = to_subid[my_id]

        dict_ = action.effect_on(substation_id=my_sub_id)
        return dict_, to_sub_pos, my_id, my_sub_id

    def set_bus(self, name_element, new_bus, extremity=None, substation=None, type_element=None, previous_action=None):
        """
        Utilities to set the bus of a single element if you give its name. **NB** Setting a bus has the effect to
        assign the object to this bus. If it was before that connected to bus 1, and you assign it to bus 1 (*new_bus*
        = 1) it will stay on bus 1. If it was on bus 2 (and you still assign it to bus 1) it will be moved to bus 2.
        1. It should not be mixed up with :func:`HelperAction.change_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (in place) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus

        new_bus: ``int``
            Id of the new bus to connect the object to.

        extremity: ``str``
            "or" or "ext" for origin or extremity, ignored if the element is not a powerline.

        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise, the method will search for it.

        type_element: ``str``, optional
            Type of the element to look for. It is here to speed up the computation. One of "line", "gen" or "load"

        previous_action: :class:`Action`, optional
            The (optional) action to update. It should be of the same type as :attr:`HelperAction.actionClass`

        Returns
        -------
        res: :class:`Action`
            The action with the modification implemented

        Raises
        ------
        AmbiguousAction
            If *previous_action* has not the same type as :attr:`HelperAction.actionClass`.

        """
        if previous_action is None:
            res = self.actionClass(gridobj=self)
        else:
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation,
                                                                        type_element, res)
        dict_["set_bus"][to_sub_pos[my_id]] = new_bus
        res.update({"set_bus": {"substations_id": [(my_sub_id, dict_["set_bus"])]}})
        return res

    def get_set_line_status_vect(self):
        """
        Computes and returns a vector that can be used in the "set_status" keyword if building an :class:`Action`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.int
            A vector that doesn't affect the grid, but can be used in "set_line_status"

        """
        return self._template_act.get_set_line_status_vect()

    def get_change_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "change_line_status" keyword if building an :class:`Action`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.bool
            A vector that doesn't affect the grid, but can be used in "change_line_status"

        """
        return self._template_act.get_change_line_status_vect()

    @staticmethod
    def get_all_unitary_topologies_change(action_space):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "change_bus" method. The "do nothing" action will be counted only
        once.

        Parameters
        ----------
        action_space: :class:`grid2op.Action.ActionHelper`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        """
        res = []
        S = [0, 1]
        for sub_id, num_el in enumerate(action_space.sub_info):
            if num_el < 4:
                pass

            for tup in itertools.product(S, repeat=num_el - 1):
                indx = np.full(shape=num_el, fill_value=False, dtype=np.bool)
                tup = np.array((0, *tup)).astype(np.bool)  # add a zero to first element -> break symmetry
                indx[tup] = True
                if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                    # i need 2 elements on each bus at least
                    action = action_space({"change_bus": {"substations_id": [(sub_id, indx)]}})
                    res.append(action)
        return res

    @staticmethod
    def get_all_unitary_topologies_set(action_space):
        """
        This methods allows to compute and return all the unitary topological changes that can be performed on a
        powergrid.

        The changes will be performed using the "set_bus" method. The "do nothing" action will be counted once
        per substation in the grid.

        Parameters
        ----------
        action_space: :class:`grid2op.Action.ActionHelper`
            The action space used.

        Returns
        -------
        res: ``list``
            The list of all the topological actions that can be performed.

        """
        res = []
        S = [0, 1]
        for sub_id, num_el in enumerate(action_space.sub_info):
            tmp = []
            new_topo = np.full(shape=num_el, fill_value=1, dtype=np.int)
            # perform the action "set everything on bus 1"
            action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
            tmp.append(action)

            powerlines_or_id = action_space.line_or_to_sub_pos[action_space.line_or_to_subid == sub_id]
            powerlines_ex_id = action_space.line_ex_to_sub_pos[action_space.line_ex_to_subid == sub_id]
            powerlines_id = np.concatenate((powerlines_or_id, powerlines_ex_id))

            # computes all the topologies at 2 buses for this substation
            for tup in itertools.product(S, repeat=num_el - 1):
                indx = np.full(shape=num_el, fill_value=False, dtype=np.bool)
                tup = np.array((0, *tup)).astype(np.bool)  # add a zero to first element -> break symmetry
                indx[tup] = True
                if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                    # i need 2 elements on each bus at least
                    new_topo = np.full(shape=num_el, fill_value=1, dtype=np.int)
                    new_topo[~indx] = 2

                    if np.sum(indx[powerlines_id]) == 0 or np.sum(~indx[powerlines_id]) == 0:
                        # if there is a "node" without a powerline, the topology is not valid
                        continue

                    action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
                    tmp.append(action)

            if len(tmp) >= 2:
                # if i have only one single topology on this substation, it doesn't make any action
                # i cannot change the topology is there is only one.
                res += tmp

        return res


class HelperAction(SerializableActionSpace):
    """
    :class:`HelperAction` should be created by an :class:`grid2op.Environment.Environment`
    with its parameters coming from a properly
    set up :class:`grid2op.Backend.Backend` (ie a Backend instance with a loaded powergrid.
    See :func:`grid2op.Backend.Backend.load_grid` for
    more information).

    It will allow, thanks to its :func:`HelperAction.__call__` method to create valid :class:`Action`. It is the
    the preferred way to create an object of class :class:`Action` in this package.

    On the contrary to the :class:`Action`, it is NOT recommended to overload this helper. If more flexibility is
    needed on the type of :class:`Action` created, it is recommended to pass a different "*actionClass*" argument
    when it's built. Note that it's mandatory that the class used in the "*actionClass*" argument derived from the
    :class:`Action`.

    Attributes
    ----------
    game_rules: :class:`grid2op.GameRules.GameRules`
        Class specifying the rules of the game used to check the legality of the actions.


    """
    
    def __init__(self, gridobj, legal_action, actionClass=Action):
        """
        All parameters (name_gen, name_load, name_line, sub_info, etc.) are used to fill the attributes having the
        same name. See :class:`HelperAction` for more information.

        Parameters
        ----------

        gridobj: :class:`grid2op.Space.GridObjects`
            The representation of the powergrid.

        actionClass: ``type``
            Note that this parameter expected a class and not an object of the class. It is used to return the
            appropriate action type.

        legal_action: :class:`grid2op.GameRules.LegalAction`
            Class specifying the rules of the game used to check the legality of the actions.

        """
        SerializableActionSpace.__init__(self, gridobj, actionClass=actionClass)
        self.legal_action = legal_action

    def __call__(self, dict_=None, check_legal=False, env=None):
        """
        This utility allows you to build a valid action, with the proper sizes if you provide it with a valid
        dictionnary.

        More information about this dictionnary can be found in the :func:`Action.update` help. This dictionnary
        is not changed in this method.

        **NB** This is the only recommended way to make a valid, with proper dimension :class:`Action` object:

        Examples
        --------
        Here is a short example on how to make a action. For more detailed examples see :func:`Action.update`

        .. code-block:: python

            import grid2op
            # create a simple environment
            env = grid2op.make()
            act = env.action_space({})
            # act is now the "do nothing" action, that doesn't modify the grid.

        Parameters
        ----------
        dict_ : :class:`dict`
            see :func:`Action.__call__` documentation for an extensive help about this parameter

        check_legal: :class:`bool`
            is there a test performed on the legality of the action. **NB** When an object of class :class:`Action` is
            used, it is automatically tested for ambiguity. If this parameter is set to ``True`` then a legality test
            is performed. An action can be illegal if the environment doesn't allow it, for example if an agent tries
            to reconnect a powerline during a maintenance.

        env: :class:`grid2op.Environment`, optional
            An environment used to perform a legality check.

        Returns
        -------
        res: :class:`Action`
            An action that is valid and corresponds to what the agent want to do with the formalism defined in
            see :func:`Action.udpate`.

        """

        res = self.actionClass(gridobj=self)
        # update the action
        res.update(dict_)
        if check_legal:
            if not self._is_legal(res, env):
                raise IllegalAction("Impossible to perform action {}".format(res))

        return res

    def _is_legal(self, action, env):
        """

        Parameters
        ----------
        action
        env

        Returns
        -------
        res: ``bool``
            ``True`` if the action is legal, ie is allowed to be performed by the rules of the game. ``False``
            otherwise.
        """
        if env is None:
            warnings.warn("Cannot performed legality check because no environment is provided.")
            return True
        return self.legal_action(action, env)
