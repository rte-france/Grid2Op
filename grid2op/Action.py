"""
The "Action" module lets you define some actions on the underlying power _grid.
These actions are either made by an agent, or by the environment.

For now, the actions can act on:

  - the "injections" and allows you to change:

    - the generators active power production setpoint
    - the generators voltage magnitude setpoint
    - the loads active power consumption
    - the loads reactive power consumption

  - the status of the powerlines (connected / disconnected)
  - the configuration at substations eg setting different object to different bus for example

The Action class is an abstract class. You can implement it the way you want. If you decide to extend it, make sure
that the :class:`grid2op.Backend` class will be able to understand it. If you don't, your extension will have no effect
on the unerlying powergrid. Indeed a :class:`grid2op.Backend` will call the :func:`Action.__call__` method and should
understands its return type.

In this module we derived two action class:

  - :class:`Action` represents an type of action that can act on all the above mentionned objects
  - :class:`TopologyAction` restricts the modification to line status modification and bus reconfiguration at substations.


The :class:`Action` and all its derivatives also offer some usefull inspection utilities:

  - :func:`Action.__str__` prints the action in a format that gives usefull information on how it will affect the powergrid
  - :func:`Action.effect_on` returns a dictionnary that gives information about its effect.

Finally, :class:`Action` class define some strict behaviour to follow if reimplementing them. The correctness of each
instances of Action is assessed both when calling :func:`Action.update` or with a call to
:func:`Action._check_for_ambiguity` performed for example by the Backend when it must implement its effect on the
powergrid through a call to :func:`Action.__call__`

"""
import numpy as np
import warnings
import copy

import pdb

try:
    from .Exceptions import *
except ModuleNotFoundError:
    from Exceptions import *
except ImportError:
    from Exceptions import *

# TODO code "reduce" multiple action (eg __add__ method, carefull with that... for example "change", then "set" is not
# ambiguous at all, same with "set" then "change")

# TODO code "json" serialization
# TODO code "convert_for" and "convert_from" to be able to change the backend (should be handled by the backend directly)
# TODO have something that output a dict like "i want to change this element" (with a simpler API than the update stuff)
# TODO time delay somewhere (eg action is implemented after xxx timestep, and not at the time where it's proposed)

# TODO have the "reverse" action, that does the opposite of an action. Will be hard but who know ? :eyes:
# TODO add serialization of ActionSpace to json or yaml

class Action(object):
    """
    This is a base class for each :class:`Action` objects.
    As stated above, an action represents in a convenient way the modifications that will affect a powergrid.

    It is not recommended to instanciate an action from scratch. The recommended way to get an action is either by
    modifying an existing one using the method :func:`Action.update` or to call and :class:`HelperAction` object that
    has been properly set up by an :class:`grid2op.Environment`.

    Action can be fully convert to and back from a numpy array with a **fixed** size.

    An action can modify the _grid in multiple way.
    It can change :

    - the production and voltage setpoint of the generator units
    - the amount of power consumed (for both active and reactive part) for load
    - disconnect powerlines
    - change the topology of the _grid.

    In order to be valid, an action should be convertible to a tuple of 5 elements:

    - the first element are the "injections": representing the way generator units and loads are modified
        - It is in turn a dictionnary with the following keys (optional)

            - "load_p" a vector of the same size of the load, giving the modification of the loads active consumption
            - "load_q" a vector of the same size of the load, giving the modification of the loads reactive consumption
            - "prod_p" a vector of the same size of the generators, giving the modification of the productions active setpoint production
            - "prod_v" a vector of the same size of the generators, giving the modification of the productions voltage setpoint

    - the second element is made of force line status. It is made of a vector of size :attr:`Action._n_lines`
      (the number of lines in the powergrid) and is interepreted as:

            - -1 force line disconnection
            - +1 force line reconnection
            - 0 do nothing to this line

    - the third element is the switch line status vector. It is made of a vector of size :attr:`Action._n_lines` and is
      interpreted as:

        - True: change the line status
        - False: don't do anything

    - the fourth element set the buses to which the object is connected. It's a vector of integer with the following
      interpretation:

        - 0 -> don't change
        - 1 -> connect to bus 1
        - 2 -> connect to bus 2
        - -1 -> disconnect the object.

    - the fifth element change the buses to which the object is connected. It's a boolean vector interpreted as:
        - False: nothing is done
        - True: change the bus eg connect it to bus 1 if it was connected to bus 2 or connect it to bus 2 if it was
          connected to bus 1. NB this is only active if the system has only 2 buses per substation (that's the case for
          the L2RPN challenge).

    **NB** the difference between :attr:`Action._set_topo_vect` and :attr:`Action._change_bus_vect` is the following:

        - If  a component of :attr:`Action._set_topo_vect` is 1, then the object (load, generator or powerline)
          will be moved to bus 1 of the substation to
          which it is connected. If it is already to bus 1 nothing will be done. If it's on another bus it will connect
          it to bus 1. It's it's disconnected, it will reconnect it and connect it to bus 1.
        - If a component of :attr:`Action._change_bus_vect` is True, then object will be moved from one bus to another.
          If the object were on bus 1
          it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
          disconnected, then this does nothing.

    The conversion to the action into a understandable format by the backend is performed by the "update" method,
    that takes into account a dictionnary, and is responsible to convert it into this format.
    It is possible to overload this class as long as the overloaded :func:`Action.__call__` operator returns the
    specified format, and the :func:`Action.__init__` method have the same signature.

    This format is then digested by the backend and the powergrid is modified accordingly.

    Attributes
    ----------
    _n_lines: :class:`int`
        number of powerline in the _grid

    _n_gen: :class:`int`
        number of generators in the _grid

    _n_load: :class:`int`
        number of loads in the powergrid

    _subs_info: :class:`numpy.array`, dtype:int
        for each substation, gives the number of elements connected to it

    _dim_topo: :class:`int`
        size of the topology vector.

    _load_to_subid: :class:`numpy.array`, dtype:int
        for each load, gives the id the substation to which it is connected

    _gen_to_subid: :class:`numpy.array`, dtype:int
        for each generator, gives the id the substation to which it is connected

    _lines_or_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "origin" end is connected

    _lines_ex_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "extremity" end is connected

    _load_to_sub_pos: :class:`numpy.array`, dtype:int
        The topology if of the subsation *i* is given by a vector, say *sub_topo_vect* of size
        :attr:`Action._subs_info`\[i\]. For a given load of id *l*, :attr:`Action._load_to_sub_pos`\[l\] is the index
        of the load *l* in the vector *sub_topo_vect*. This means that, if
        *sub_topo_vect\[ action._load_to_sub_pos\[l\] \]=2*
        then load of id *l* is connected to the second bus of the substation.

    _gen_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Action._load_to_sub_pos` but for generators.

    _lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Action._load_to_sub_pos` but for "origin" end of powerlines.

    _lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Action._load_to_sub_pos` but for "extremity" end of powerlines.

    _load_pos_topo_vect: :class:`numpy.array`, dtype:int
        It has a similar role as :attr:`Action._load_to_sub_pos` but it gives the position in the vector representing
        the whole topology. More concretely, if the complete topology of the powergrid is represented here by a vector
        *full_topo_vect* resulting of the concatenation of the topology vector for each substation
        (see :attr:`Action._load_to_sub_pos`for more information). For a load of id *l* in the powergrid,
        :attr:`Action._load_pos_topo_vect`\[l\] gives the index, in this *full_topo_vect* that concerns load *l*.
        More formally, if *_topo_vect\[ action._load_pos_topo_vect\[l\] \]=2* then load of id l is connected to the
        second bus of the substation.

    _gen_pos_topo_vect: :class:`numpy.array`, dtype:int
         same as :attr:`Action._load_pos_topo_vect` but for generators.

    _lines_or_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Action._load_pos_topo_vect` but for "origin" end of powerlines.

    _lines_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Action._load_pos_topo_vect` but for "extremity" end of powerlines.

    _set_line_status: :class:`numpy.array`, dtype:int
        For each powerlines, it gives the effect of the action on the status of it. It should be understand as:

          - -1 : disconnect the powerline
          - 0 : don't affect the powerline
          - +1 : reconnect the powerline

    _switch_line_status: :class:`numpy.array`, dtype:bool
        For each powerline, it informs whether the action will switch the status of a powerline of not. It should be
        understood as followed:

          - ``False`` : the action doesn't affect the powerline
          - ``True`` : the action affect the powerline. If it was connected, it will disconnect it. If it was
            disconnected, it will reconnect it.

    _dict_inj: ``dict``
        Represents the modification of the injection (productions and loads) of the power _grid. This dictionnary can
        have the optional keys:

            - "load_p" to set the active load values (this is a np array with the same size as the number of load
                in the power _grid with Nan: don't change anything, else set the value
            - "load_q" : same as above but for the load reactive values
            - "prod_p" : same as above but for the generator active setpoint values. It has the size corresponding
                to the number of generators in the test case.
            - "prod_v" : same as above but set the voltage setpoint of generator units.

    _set_topo_vect: :class:`numpy.array`, dtype:int
        Similar to :attr:`Action._set_line_status` but instead of affecting the status of powerlines, it affects the
        bus connectivity at substation. It has the same size as the full topological vector (:attr:`Action._dim_topo`)
        and for each element it should be understood as:

            - 0 : nothing is changed for this element
            - +1 : this element is affected to bus 1
            - -1 : this element is affected to bus 2

    _change_bus_vect: :class:`numpy.array`, dtype:bool
         Similar to :attr:`Action._switch_line_status` but it affects the topology at substations instead of the status
         of the powerline. It has the same size as the full topological vector (:attr:`Action._dim_topo`) and each
         component should means:

             - ``False`` : the object is not affected
             - ``True`` : the object will be moved to another bus. If it was on bus 1 it will be moved on bus 2, and if
                 it was on bus 2 it will be moved on bus 1.

    authorized_keys: :class:`set`
        The set indicating which keys the actions is able to understand when calling :func:`Action.update`

    as_vect: :class:`numpy.array`, dtype:float
        The representation of the action as a vector. See the help of :func:`Action.to_vect` and
        :func:`Action.from_vect` for more information. **NB** for performance reason, the convertion of the internal
        representation to a vector is not performed at any time. It is only performed when :func:`Action.to_vect` is
        called. Otherwise, this attribute is set to ``None``

    _subs_impacted: :class:`numpy.array`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each substation, if it is impacted
        by the action (in this case :attr:`Action._subs_impacted`\[sub_id\] is ``True``) or not
        (in this case :attr:`Action._subs_impacted`\[sub_id\] is ``False``)

    _lines_impacted: :class:`numpy.array`, dtype:bool
        This attributes is either not initialized (set to ``None``) or it tells, for each powerline, if it is impacted
        by the action (in this case :attr:`Action._lines_impacted`\[line_id\] is ``True``) or not
        (in this case :attr:`Action._subs_impacted`\[line_id\] is ``False``)

    vars_action: ``list``, static
        Authorized key that are processed by :func:`Action.__call__` to modify the injections

    vars_action_set: ``set``, static
        Authorized key that are processed by :func:`Action.__call__` to modify the injections
    """

    vars_action = ["load_p", "load_q", "prod_p", "prod_v"]
    vars_action_set = set(vars_action)

    def __init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect):
        """
        This is used to create an Action instance. Preferably, :class:`Action` should be created with
        :class:`HelperAction`.

        **It is NOT recommended** to create an action with this method. Please use :func:`HelperAction.__call__` or
        :func:`HelperAction.sample` to create a valid action.

        Parameters
        ----------
        n_gen
            Use to initialize :attr:`Action._n_gen`.

        n_load
            Use to initialize :attr:`Action._n_load`.

        n_lines
            Use to initialize :attr:`Action._n_lines`.

        subs_info
            Use to initialize :attr:`Action._subs_info`.

        dim_topo
            Use to initialize :attr:`Action._dim_topo`.

        load_to_subid
            Use to initialize :attr:`Action._load_to_subid`.

        gen_to_subid
            Use to initialize :attr:`Action._gen_to_subid`.

        lines_or_to_subid
            Use to initialize :attr:`Action._lines_or_to_subid`.

        lines_ex_to_subid
            Use to initialize :attr:`Action._lines_ex_to_subid`.

        load_to_sub_pos
            Use to initialize :attr:`Action._load_to_sub_pos`.

        gen_to_sub_pos
            Use to initialize :attr:`Action._gen_to_sub_pos`.

        lines_or_to_sub_pos
            Use to initialize :attr:`Action._lines_or_to_sub_pos`.

        lines_ex_to_sub_pos
            Use to initialize :attr:`Action._lines_ex_to_sub_pos`.

        load_pos_topo_vect
            Use to initialize :attr:`Action._load_pos_topo_vect`.

        gen_pos_topo_vect
            Use to initialize :attr:`Action._gen_pos_topo_vect`.

        lines_or_pos_topo_vect
            Use to initialize :attr:`Action._lines_or_pos_topo_vect`.

        lines_ex_pos_topo_vect
            Use to initialize :attr:`Action._lines_ex_pos_topo_vect`.

        """

        self._n_gen = n_gen
        self._n_load = n_load
        self._n_lines = n_lines
        self._subs_info = subs_info
        self._dim_topo = dim_topo

        # to which substation is connected each element
        self._load_to_subid = load_to_subid
        self._gen_to_subid = gen_to_subid
        self._lines_or_to_subid = lines_or_to_subid
        self._lines_ex_to_subid = lines_ex_to_subid
        # which index has this element in the substation vector
        self._load_to_sub_pos = load_to_sub_pos
        self._gen_to_sub_pos = gen_to_sub_pos
        self._lines_or_to_sub_pos = lines_or_to_sub_pos
        self._lines_ex_to_sub_pos = lines_ex_to_sub_pos
        # which index has this element in the topology vector
        self._load_pos_topo_vect = load_pos_topo_vect
        self._gen_pos_topo_vect = gen_pos_topo_vect
        self._lines_or_pos_topo_vect = lines_or_pos_topo_vect
        self._lines_ex_pos_topo_vect = lines_ex_pos_topo_vect

        self.authorized_keys = {"injection",
                                "hazards", "maintenance", "set_line_status", "change_line_status",
                                "set_bus", "change_bus"}

        # False(line is disconnected) / True(line is connected)
        self._set_line_status = np.full(shape=n_lines, fill_value=0, dtype=np.int)
        self._switch_line_status = np.full(shape=n_lines, fill_value=False, dtype=np.bool)

        # injection change
        self._dict_inj = {}

        # topology changed
        self._set_topo_vect = np.full(shape=self._dim_topo, fill_value=0, dtype=np.int)
        self._change_bus_vect = np.full(shape=self._dim_topo, fill_value=False, dtype=np.bool)

        self.as_vect = None

        self._subs_impacted = None
        self._lines_impacted = None

    def get_set_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "set_status" keyword if building an :class:`Action`.

        **NB** this vector is not the internal vector of this action, but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.int
            A vector that doesn't affect the grid, but can be used in "set_status"

        """
        return np.full(shape=self._n_lines, fill_value=0, dtype=np.int)

    def get_change_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "change_status" keyword if building an :class:`Action`

        **NB** this vector is not the internal vector of this action, but corresponds to "do nothing" action.

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.bool
            A vector that doesn't affect the grid, but can be used in "change_status"

        """
        return np.full(shape=self._n_lines, fill_value=False, dtype=np.bool)

    def __eq__(self, other) -> bool:
        """
        Test the equality of two actions.

        2 actions are said to be identical if the have the same impact on the powergrid. This is unrelated to their
        respective class. For example, if an Action is of class :class:`Action` and doesn't act on the injection, it
        can be equal to a an Action of derived class :class:`TopologyAction` (if the topological modification are the
        same of course).

        This implies that the attributes :attr:`Action.authorized_keys` is not checked in this method.

        Note that if 2 actions doesn't act on the same powergrid, or on the same backend (eg number of loads, or
        generators is not the same in *self* and *other*, or they are not in the same order) then action will be
        declared as different.

        **Known issue** if two backend are different, but the description of the _grid are identical (ie all
        _n_gen, _n_load, _n_lines, _subs_info, _dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behaviour: if everything is the same, then probably, up to
        the naming convention, then the powergrid are identical too.

        Parameters
        ----------
        other: :class:`Action`
            An instance of class Action to which "self" will be compared.

        Returns
        -------

        """

        # check that the _grid is the same in both instances
        same_grid = True
        same_grid = same_grid and self._n_gen == other._n_gen
        same_grid = same_grid and self._n_load == other._n_load
        same_grid = same_grid and self._n_lines == other._n_lines
        same_grid = same_grid and np.all(self._subs_info == other._subs_info)
        same_grid = same_grid and self._dim_topo == other._dim_topo
        # to which substation is connected each element
        same_grid = same_grid and np.all(self._load_to_subid == other._load_to_subid)
        same_grid = same_grid and np.all(self._gen_to_subid == other._gen_to_subid)
        same_grid = same_grid and np.all(self._lines_or_to_subid == other._lines_or_to_subid)
        same_grid = same_grid and np.all(self._lines_ex_to_subid == other._lines_ex_to_subid)
        # which index has this element in the substation vector
        same_grid = same_grid and np.all(self._load_to_sub_pos == other._load_to_sub_pos)
        same_grid = same_grid and np.all(self._gen_to_sub_pos == other._gen_to_sub_pos)
        same_grid = same_grid and np.all(self._lines_or_to_sub_pos == other._lines_or_to_sub_pos)
        same_grid = same_grid and np.all(self._lines_ex_to_sub_pos == other._lines_ex_to_sub_pos)
        # which index has this element in the topology vector
        same_grid = same_grid and np.all(self._load_pos_topo_vect == other._load_pos_topo_vect)
        same_grid = same_grid and np.all(self._gen_pos_topo_vect == other._gen_pos_topo_vect)
        same_grid = same_grid and np.all(self._lines_or_pos_topo_vect == other._lines_or_pos_topo_vect)
        same_grid = same_grid and np.all(self._lines_ex_pos_topo_vect == other._lines_ex_pos_topo_vect)
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
              disconnected by a hazards at the same time step, then this action will not be implemented on the grid.
              However, it this powerline couldn't be reconnected for some reason (for example it was already out of
              order) the action will still be declared illegal, even if it has NOT impacted the powergrid.
            - If an action tries to disconnect a powerline already disconnected, it will "impact" this powergrid. This
              means that even if the action will basically do nothing, it disconnecting this powerline is against the
              rules, then the action will be illegal.
            - If an action tries to change the topology of a substation, but this substation is already at the target
              topology, the same mechanism applies. The action will "impact" the substation, even if, at the end, it
              consists of doing nothing.

        Any such "change" that would be illegal are declared as "illegal" regardless of the real impact of this action
        on the powergrid.


        Returns
        -------
        lines_impacted: :class:`numpy.array`, dtype:np.bool
            A vector with the same size as the number of powerline in the grid (:attr:`Action._n_lines`) with for each
            component ``True`` if the line STATUS is impacted by the action, and ``False`` otherwise. See
            :attr:`Action._lines_impacted` for more information.

        subs_impacted: :class:`numpy.array`, dtype:np.bool
            A vector with the same size as the number of substations in the grid with for each
            component ``True`` if the substation is impacted by the action, and ``False`` otherwise. See
            :attr:`Action._subs_impacted` for more information.

        """
        if self._subs_impacted is None:
            self._subs_impacted = np.full(shape=self._subs_info.shape, fill_value=False, dtype=np.bool)
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self._subs_info):
                nb_obj = int(nb_obj)
                end_ += nb_obj
                if np.any(self._change_bus_vect[beg_:end_]) or np.any(self._set_topo_vect[beg_:end_] != 0):
                    self._subs_impacted[sub_id] = True
                beg_ += nb_obj

        if self._lines_impacted is None:
            self._lines_impacted = self._switch_line_status | (self._set_line_status != 0)
        return self._lines_impacted, self._subs_impacted

    def reset(self):
        """
        Reset the action to the "do nothing" state.
        Returns
        -------

        """
        # False(line is disconnected) / True(line is connected)
        self._set_line_status = np.full(shape=self._n_lines, fill_value=0, dtype=np.int)
        self._switch_line_status = np.full(shape=self._n_lines, fill_value=False, dtype=np.bool)

        # injection change
        self._dict_inj = {}

        # topology changed
        self._set_topo_vect = np.full(shape=self._dim_topo, fill_value=0, dtype=np.int)
        self._change_bus_vect = np.full(shape=self._dim_topo, fill_value=False, dtype=np.bool)

        self.as_vect = None

    def __call__(self):
        """
        This method is used to return the effect of the current action in a format understandable by the backend.
        This format is detailed bellow.

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

        Raises
        -------
        AmbiguousAction
            Or one of its derivate class.
        """
        self._check_for_ambiguity()
        return self._dict_inj, self._set_line_status, self._switch_line_status,\
               self._set_topo_vect, self._change_bus_vect

    def _digest_injection(self, dict_):
        # I update the action
        if "injection" in dict_:
            if dict_["injection"] is not None:
                tmp_d = dict_["injection"]
                for k in tmp_d: #["load_p", "prod_p", "load_q", "prod_v"]:
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
                        if c_id >= self._n_lines:
                            raise AmbiguousAction("Load {} doesn't exist".format(c_id))
                        self._set_topo_vect[self._load_pos_topo_vect[c_id]] = bus
                        # print("self._load_pos_topo_vect[l_id] {}".format(self._load_pos_topo_vect[l_id]))
                if "generators_id" in ddict_:
                    tmp = ddict_["generators_id"]
                    handled = True
                    for (g_id, bus) in tmp:
                        if g_id >= self._n_gen:
                            raise AmbiguousAction("Generator {} doesn't exist".format(g_id))
                        self._set_topo_vect[self._gen_pos_topo_vect[g_id]] = bus
                if "lines_or_id" in ddict_:
                    tmp = ddict_["lines_or_id"]
                    handled = True
                    for (l_id, bus) in tmp:
                        if l_id >= self._n_lines:
                            raise AmbiguousAction("Powerline {} doesn't exist".format(l_id))
                        self._set_topo_vect[self._lines_or_pos_topo_vect[l_id]] = bus
                if "lines_ex_id" in ddict_:
                    tmp = ddict_["lines_ex_id"]
                    handled = True
                    for (l_id, bus) in tmp:
                        if l_id >= self._n_lines:
                            raise AmbiguousAction("Powerline {} doesn't exist".format(l_id))
                        self._set_topo_vect[self._lines_ex_pos_topo_vect[l_id]] = bus
                if "substations_id" in ddict_:
                    handled = True
                    tmp = ddict_["substations_id"]
                    for (s_id, arr) in tmp:
                        if s_id >= self._subs_info.shape[0]:
                            raise AmbiguousAction("Substation {} doesn't exist".format(s_id))

                        s_id = int(s_id)
                        beg_ = int(np.sum(self._subs_info[:s_id]))
                        end_ = int(beg_ + self._subs_info[s_id])
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
                        self._change_bus_vect[self._load_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self._load_pos_topo_vect[l_id]]
                if "generators_id" in ddict_:
                    tmp = ddict_["generators_id"]
                    for g_id in tmp:
                        self._change_bus_vect[self._gen_pos_topo_vect[g_id]] = not self._change_bus_vect[
                            self._gen_pos_topo_vect[g_id]]
                if "lines_or_id" in ddict_:
                    tmp = ddict_["lines_or_id"]
                    for l_id in tmp:
                        self._change_bus_vect[self._lines_or_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self._lines_or_pos_topo_vect[l_id]]
                if "lines_ex_id" in ddict_:
                    tmp = ddict_["lines_ex_id"]
                    for l_id in tmp:
                        self._change_bus_vect[self._lines_ex_pos_topo_vect[l_id]] = not self._change_bus_vect[
                            self._lines_ex_pos_topo_vect[l_id]]
                if "substations_id" in ddict_:
                    tmp = ddict_["substations_id"]
                    for (s_id, arr) in tmp:
                        s_id = int(s_id)
                        beg_ = int(np.sum(self._subs_info[:s_id]))
                        end_ = int(beg_ + self._subs_info[s_id])
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
                    if len(dict_["set_line_status"]) != self._n_lines:
                        raise InvalidNumberOfLines(
                            "This \"set_line_status\" action acts on {} lines while there are {} in the grid".format(
                                len(dict_["set_line_status"]), self._n_lines))
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
                    if len(tmp) != self._n_lines:
                        raise InvalidNumberOfLines(
                            "This \"hazards\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self._n_lines))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction("You can only ask hazards with int or boolean numpy array vector.")

                # if len(dict_["outage"]) != self._n_lines:
                #     raise InvalidNumberOfLines("This action acts on {} lines while there are {} in the _grid".format(len(dict_["outage"]), self._n_lines))
                self._set_line_status[tmp] = -1
                # force ignore of any topological actions
                self._ignore_topo_action_if_disconnection(tmp)

    def _digest_maintenance(self, dict_):
        if "maintenance" in dict_:
            # set the values of the power lines to "disconnected" for element being "False"
            # does nothing to the others
            # a _maintenance operation will never reconnect a powerline

            # if len(dict_["_maintenance"]) != self._n_lines:
            #     raise InvalidNumberOfLines("This action acts on {} lines while there are {} in the _grid".format(len(dict_["_maintenance"]), self._n_lines))

            if dict_["maintenance"] is not None:
                tmp = dict_["maintenance"]
                try:
                    tmp = np.array(tmp)
                except:
                    raise AmbiguousAction(
                        "You ask to perform maintenance on powerlines, this can only be done if \"maintenance\" is castable into a numpy ndarray")
                if np.issubdtype(tmp.dtype, np.dtype(bool).type):
                    if len(tmp) != self._n_lines:
                        raise InvalidNumberOfLines(
                            "This \"maintenance\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self._n_lines))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction(
                        "You can only ask to perform lines maintenance with int or boolean numpy array vector.")
                self._set_line_status[tmp] = -1
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
                    if len(tmp) != self._n_lines:
                        raise InvalidNumberOfLines(
                            "This \"change_line_status\" action acts on {} lines while there are {} in the _grid".format(
                                len(tmp), self._n_lines))
                elif not np.issubdtype(tmp.dtype, np.dtype(int).type):
                    raise AmbiguousAction("You can only change line status with int or boolean numpy array vector.")
                self._switch_line_status[dict_["change_line_status"]] = True

    def update(self, dict_):
        """
        Update the action with a comprehensible format specified by a dictionnary.

        Preferably, if a keys of the argument *dict_* is not found in :attr:`Action.authorized_keys` it should throw a
        warning. This argument will be completely ignored.

        This method also reset the attributes :attr:`Action.as_vect` :attr:`Action._lines_impacted` and
        :attr:`Action._subs_impacted` to ``None`` regardless of the argument in input.

        If an action consist in "reconnecting" a powerline, and this same powerline is affected by a maintenance or a
        hazard, it will be erased without any warning. "hazards" and "maintenance" have the priority. This is a
        requirements for all proper :class:`Action` subclass.

        Parameters
        ----------
        dict_: :class:`dict`
            If it's ``None`` or empty it does nothing. Otherwise, it can contain the following (optional) keys:

            - "*injection*" if the action will modify the injections (generator setpoint / load value - active or
              reactive) of the powergrid. It has optionally one of the following keys:

                    - "load_p" to set the active load values (this is a np array with the same size as the number of
                      load in the power _grid with Nan: don't change anything, else set the value
                    - "load_q" : same as above but for the load reactive values
                    - "prod_p" : same as above but for the generator active setpoint values. It has the size
                      corresponding to the number of generators in the test case.
                    - "prod_v" : same as above but set the voltage setpoint of generator units.

            - "*hazards*": represents the hazards that the line might suffer (boolean vector) False: no hazard, nothing
              is done, True: an hazard, the powerline is disconnected
            - "*maintenance*": represents the maintenance operation performed on each powerline (boolean vector) False:
              no maintenance, nothing is done, True: a maintenance is scheduled, the powerline is disconnected
            - "*set_line_status*": a vector (int or float) to set the status of the powerline status (connected /
              disconnected) with the following interpretation:

                - 0 : nothing is changed,
                - -1 : disconnect the powerline,
                - +1 : reconnect the powerline. If an action consist in "reconnect" a powerline, and this same
                  powerline is affected by a maintenance or a hazard, it will be erased without any warning. "hazards"
                  and "maintenance" have the priority.

            - "change_line_status": a vector (bool) to change the status of the powerline. This vector should be interpreted
              as:

                - False: do nothing
                - True: change the status of the powerline: disconnect it if it was connected, connect it if it was
                  disconnected

            - "set_bus": (numpy int vector or dict) will set the buses to which the objects is connected. It follows a
              similar interpretation than the line status vector:

                - 0 -> don't change anything
                - +1 -> set to bus 1,
                - +2 -> set to bus 2, etc.
                - -1: You can use this method to disconnect an object by setting the value to -1.

            - "change_bus": (numpy bool vector or dict) will change the bus to which the object is connected. True will
              change it (eg switch it from bus 1 to bus 2 or from bus 2 to bus 1). NB this is only active if the system
              has only 2 buses per substation.

            **NB** the difference between "set_bus" and "change_bus" is the following:

              - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the
                substation to which it is connected. If it is already to bus 1 nothing will be done. If it's on another
                bus it will connect it to bus 1. It's it's disconnected, it will reconnect it and connect it to bus 1.
              - If "change_bus" is True, then object will be moved from one bus to another. If the object where on bus 1
                then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object is
                disconnected then the action is ambiguous, and calling it will throw an AmbiguousAction exception.

            **NB**: if a powerline is reconnected, it should be specified on the "set_bus" action at which buses it
            should be  reconnected. Otherwise, action cannot be used. Trying to apply the action to the _grid will
            lead to a "AmbiguousAction" exception.

            **NB**: if for a given powerline, both switch_line_status and set_line_status is set, the action will not
            be usable.
            This will lead to an :class:`grid2op.Exception.AmbiguousAction` exception.

            **NB**: length of vector here are NOT check in this function. This method can be "chained" and only on the final
            action, when used, eg. in the Backend, i checked.

            **NB**: If a powerline is disconnected, on maintenance, or suffer an outage, the associated "set_bus" will
            be ignored.
            Disconnection has the priority on anything. This priority is given because in case of hazard, the hazard has
            the priority over the possible actions.

        Returns
        -------
        self: :class:`Action`
            Return the modified instance. This is handy to chain modifications if needed.

        """
        self.as_vect = None
        self._subs_impacted = None
        self._lines_impacted = None

        if dict_ is not None:
            for kk in dict_.keys():
                if kk not in self.authorized_keys:
                    warn = "The key \"{}\" used to update an action will be ignored. Valid keys are {}"
                    warn = warn.format(kk, self.authorized_keys)
                    warnings.warn(warn)

            self._digest_injection(dict_)
            self._digest_setbus(dict_)
            self._digest_change_bus(dict_)
            self._digest_set_status(dict_)
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)

        return self

    def _check_for_ambiguity(self):
        """
        This method check if an action is ambiguous or not. If the instance is ambiguous, an
        :class:`grid2op.Exceptions.AmbiguousAction` is raised.

        An action can be ambiguous in the following context:

          - It affects the injections in an incorrect way:

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

          - It has an ambiguous behaviour concerning the topology of some substations

            - the state of some bus for some element is both *changed* (:code:`self._change_bus_vect[i] = True` for
              some *i*) and *set* (:code:`self._set_topo_vect[i]` for the same *i* is not 0)
            - :code:`self._set_topo_vect` has not the same dimension as the number of elements on the powergrid
            - :code:`self._change_bus_vect` has not the same dimension as the number of elements on the powergrid

        In case of need to overload this method, it is advise to still call this one from the base :class:`Action`
        with ":code:`super()._check_for_ambiguity()`" or ":code:`Action._check_for_ambiguity(self)`".

        Returns
        -------
        ``None``

        Raises
        -------
        AmbiguousAction
            Or any of its more precise subclasses, depending on which assumption is not met.

        """
        if np.any(self._set_line_status[self._switch_line_status] != 0):
            raise InvalidLineStatus("You asked to change the status (connected / disconnected) of a powerline by"
                                    " using the keyword \"change_status\" and set this same line state in \"set_status\""
                                    " (or \"hazard\" or \"maintenance\"). This ambiguous behaviour is not supported")
        # check size
        if "load_p" in self._dict_inj:
            if len(self._dict_inj["load_p"]) != self._n_load:
                raise InvalidNumberOfLoads("This action acts on {} loads while there are {} in the _grid".format(len(self._dict_inj["load_p"]), self._n_load))
        if "load_q" in self._dict_inj:
            if len(self._dict_inj["load_q"]) != self._n_load:
                raise InvalidNumberOfLoads("This action acts on {} loads while there are {} in the _grid".format(len(self._dict_inj["load_q"]), self._n_load))
        if "prod_p" in self._dict_inj:
            if len(self._dict_inj["prod_p"]) != self._n_gen:
                raise InvalidNumberOfGenerators("This action acts on {} generators while there are {} in the _grid".format(len(self._dict_inj["prod_p"]), self._n_gen))
        if "prod_v" in self._dict_inj:
            if len(self._dict_inj["prod_v"]) != self._n_gen:
                raise InvalidNumberOfGenerators("This action acts on {} generators while there are {} in the _grid".format(len(self._dict_inj["prod_v"]), self._n_gen))

        if len(self._switch_line_status) != self._n_lines:
                raise InvalidNumberOfLines("This action acts on {} lines while there are {} in the _grid".format(len(self._switch_line_status), self._n_lines))

        if len(self._set_topo_vect) != self._dim_topo:
                raise InvalidNumberOfObjectEnds("This action acts on {} ends of object while there are {} in the _grid".format(len(self._set_topo_vect), self._dim_topo))
        if len(self._change_bus_vect) != self._dim_topo:
                raise InvalidNumberOfObjectEnds("This action acts on {} ends of object while there are {} in the _grid".format(len(self._change_bus_vect), self._dim_topo))

        if np.any(self._set_topo_vect[self._change_bus_vect] != 0):
            raise InvalidBusStatus("You asked to change the bus of an object with"
                                    " using the keyword \"change_bus\" and set this same object state in \"set_bus\""
                                    ". This ambiguous behaviour is not supported")

        for q_id, status in enumerate(self._set_line_status):
            if status == 1:
                # i reconnect a powerline, i need to check that it's connected on both ends
                if self._set_topo_vect[self._lines_or_pos_topo_vect[q_id]] == 0 or \
                        self._set_topo_vect[self._lines_ex_pos_topo_vect[q_id]] == 0:
                    raise InvalidLineStatus("You ask to reconnect powerline {} yet didn't tell on which bus.".format(q_id))

        # if i disconnected of a line, but i modify also the bus where it's connected
        idx = self._set_line_status == -1
        id_disc = np.where(idx)[0]
        if np.any(self._set_topo_vect[self._lines_or_pos_topo_vect[id_disc]] > 0) or \
                np.any(self._set_topo_vect[self._lines_ex_pos_topo_vect[id_disc]] > 0):
                    raise InvalidLineStatus("You ask to disconnect a powerline but also to connect it to a certain bus.")
        if np.any(self._change_bus_vect[self._lines_or_pos_topo_vect[id_disc]] > 0) or \
                np.any(self._change_bus_vect[self._lines_ex_pos_topo_vect[id_disc]] > 0):
                    raise InvalidLineStatus("You ask to disconnect a powerline but also to change its bus.")

        if np.any(self._change_bus_vect[self._lines_or_pos_topo_vect[self._set_line_status == 1]]):
            raise InvalidLineStatus("You ask to connect an origin powerline but also to *change* the bus  to which it is connected. This is ambiguous. You must *set* this bus instead.")
        if np.any(self._change_bus_vect[self._lines_ex_pos_topo_vect[self._set_line_status == 1]]):
            raise InvalidLineStatus("You ask to connect an extremity powerline but also to *change* the bus  to which it is connected. This is ambiguous. You must *set* this bus instead.")

    def size(self):
        """
        When an action is converted to a plain numpy array, this is the size of such an array.

        See the documentation of :func:`Action.to_vect` for more information about this array.

        If this method is overloaded, it is mandatory to overload also:

          - :func:`Action.from_vect`
          - :func:`Action.to_vect`

        Returns
        -------
        size: ``int``
            The size of the flatten array returned by :func:`Action.to_vect`.
        """
        return 2 * self._n_gen + 2 * self._n_load + 2 * self._n_lines + 2 * self._dim_topo

    def to_vect(self):
        """
        When an action is converted it to a plain numpy array, this is the size of such an array.

        All elements of all numpy array are converted to ``float``.
        By default, the order is:

          1. All modifications of generator units, with Nan if the vector is not present in the initial action

            1. :code:`self.prod_p`
            2. :code:`self.prod_v`

          2. All modifications of loads, with Nan if the vector is not present in the initial action

            1. :code:`self.load_p`
            2. :code:`self.load_v`

          3. All modifications of line status

            1. :code:`self._set_line_status`
            2. :code:`self._switch_line_status`

          4. All topological information

            1. :code:`self._set_topo_vect`
            2. :code:`self._change_bus_vect`

        If this method is overloaded, it is mandatory to overload also:

          - :func:`Action.size`
          - :func:`Action.from_vect`

        Returns
        -------
        res: :class:`numpy.array`, dtype:float
            The flatten representation of an array.

        Raises
        ------
        AmbiguousAction
            When the vector built has not the same size as a call to :func:`Action.size`.
        """
        if self.as_vect is None:
            if "prod_p" in self._dict_inj:
                prod_p = self._dict_inj["prod_p"]
            else:
                prod_p = np.full(self._n_gen, fill_value=np.NaN)
            if "prod_v" in self._dict_inj:
                prod_v = self._dict_inj["prod_v"]
            else:
                prod_v = np.full(self._n_gen, fill_value=np.NaN)

            if "load_p" in self._dict_inj:
                load_p = self._dict_inj["load_p"]
            else:
                load_p = np.full(self._n_load, fill_value=np.NaN)
            if "load_q" in self._dict_inj:
                load_q = self._dict_inj["load_q"]
            else:
                load_q = np.full(self._n_load, fill_value=np.NaN)

            self.as_vect = np.concatenate((
                prod_p.flatten().astype(np.float),
                prod_v.flatten().astype(np.float),
                load_p.flatten().astype(np.float),
                load_q.flatten().astype(np.float),
                self._set_line_status.flatten().astype(np.float),
                self._switch_line_status.flatten().astype(np.float),
                self._set_topo_vect.flatten().astype(np.float),
                self._change_bus_vect.flatten().astype(np.float)
                              ))

            if self.as_vect.shape[0] != self.size():
                raise AmbiguousAction("Action has not the proper shape.")

        return self.as_vect

    def from_vect(self, vect):
        """
        Convert a action given as a vector into a proper :class:`Action`.

        If this method is overloaded, the subclass should make sure that :func:`Action._check_for_ambiguity` is called
        after the action has been loaded.

        If this method is overloaded, it is mandatory to overload also:

          - :func:`Action.size`
          - :func:`Action.to_vect`

        Parameters
        ----------
        vect: :class:`numpy.array`, dtype:float
            The array representation of an action

        Returns
        -------
        ``None``

        Raises
        ------
        IncorrectNumberOfElements
            if the size of the vector is not the same as the result of a call to :func:`Action.size`
        """
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while load an action from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))
        prev_ = 0
        next_ = self._n_gen
        prod_p = vect[prev_:next_]; prev_ += self._n_gen; next_ += self._n_gen
        prod_q = vect[prev_:next_]; prev_ += self._n_gen; next_ += self._n_load

        load_p = vect[prev_:next_]; prev_ += self._n_load; next_ += self._n_load
        load_q = vect[prev_:next_]; prev_ += self._n_load; next_ += self._n_lines

        if np.any(np.isfinite(prod_p)):
            self._dict_inj["prod_p"] = prod_p
        if np.any(np.isfinite(prod_q)):
            self._dict_inj["prod_q"] = prod_q
        if np.any(np.isfinite(load_p)):
            self._dict_inj["load_p"] = load_p
        if np.any(np.isfinite(load_q)):
            self._dict_inj["load_q"] = load_q

        self._set_line_status = vect[prev_:next_]; prev_ += self._n_lines; next_ += self._n_lines
        self._set_line_status = self._set_line_status.astype(np.int)
        self._switch_line_status = vect[prev_:next_]; prev_ += self._n_lines; next_ += self._dim_topo
        self._switch_line_status = self._switch_line_status.astype(np.bool)
        self._set_topo_vect = vect[prev_:next_]; prev_ += self._dim_topo; next_ += self._dim_topo
        self._set_topo_vect = self._set_topo_vect.astype(np.int)
        self._change_bus_vect = vect[prev_:]; prev_ += self._dim_topo
        self._change_bus_vect = self._change_bus_vect.astype(np.bool)

        self._check_for_ambiguity()

    def sample(self):
        """
        This method is used to sample action.

        Generic sampling of action can be really tedious. Uniform sampling is almost impossible.
        The actual implementation gives absolutely no warranty toward any of these concerns. It will:
        TODO

        By calling :func:`Action.sample`, the action is :func:`Action.reset` to a "do nothing" state.
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
        self._set_topo_vect[np.array(self._lines_or_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self._lines_or_pos_topo_vect[sel_])] = False
        self._set_topo_vect[np.array(self._lines_ex_pos_topo_vect[sel_])] = 0
        self._change_bus_vect[np.array(self._lines_ex_pos_topo_vect[sel_])] = False

    def _obj_caract_from_topo_id(self, id_):
        obj_id = None
        objt_type = None
        array_subid = None
        for l_id, id_in_topo in enumerate(self._load_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                objt_type = "load"
                array_subid = self._load_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self._gen_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "generator"
                    array_subid = self._gen_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self._lines_or_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "line (origin)"
                    array_subid = self._lines_or_to_subid
        if obj_id is None:
            for l_id, id_in_topo in enumerate(self._lines_ex_pos_topo_vect):
                if id_in_topo == id_:
                    obj_id = l_id
                    objt_type = "line (extremity)"
                    array_subid = self._lines_ex_to_subid
        substation_id = array_subid[obj_id]
        return obj_id, objt_type, substation_id

    def __str__(self):
        """
        This utility allows to print in a human readable format which objects will be impacted by the action.

        Returns
        -------
        str: :class:`str`
            The string representation of an :class:`Action`
        """
        res = ["This action will:"]
        # handles actions on injections
        inj_changed = False
        for k in ["load_p", "prod_p", "load_q", "prod_q"]:
            if k in self._dict_inj:
                inj_changed = True
                res.append("\t - set {} to {}".format(k, list(self._dict_inj[k])))
        if not inj_changed:
            res.append("\t - NOT change anything to the injections")

        # handles actions on force line status
        force_linestatus_change = False
        if np.any(self._set_line_status == 1):
            res.append("\t - force reconnection of {} powerlines ({})".format(np.sum(self._set_line_status == 1),
                                                                              np.where(self._set_line_status == 1)[0]))
            force_linestatus_change = True
        if np.any(self._set_line_status == -1):
            res.append("\t - force disconnection of {} powerlines ({})".format(np.sum(self._set_line_status == -1),
                                                                               np.where(self._set_line_status == -1)[0]))
            force_linestatus_change = True
        if not force_linestatus_change:
            res.append("\t - NOT force any line status")

        # handles action on swtich line status
        if np.sum(self._switch_line_status):
            res.append("\t - switch status of {} powerlines ({})".format(np.sum(self._switch_line_status),
                                                                         np.where(self._switch_line_status)[0]))
        else:
            res.append("\t - NOT switch any line status")

        # handles topology
        if np.any(self._change_bus_vect):
            res.append("\t - Change the bus of the following element:")
            for id_, k in enumerate(self._change_bus_vect):
                if k:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    res.append("\t \t - switch bus of {} {} [on substation {}]".format(objt_type, obj_id, substation_id))
        else:
            res.append("\t - NOT switch anything in the topology")

        if np.any(self._set_topo_vect != 0):
            res.append("\t - Set the bus of the following element:")
            for id_, k in enumerate(self._set_topo_vect):
                if k > 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    res.append("\t \t - assign bus {} to {} {} [on substation {}]".format(k, objt_type, obj_id, substation_id))
                if k < 0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    res.append("\t - disconnect {} {} [on substation {}]".format(k, objt_type, obj_id, substation_id))
        else:
            res.append("\t - NOT force any particular bus configuration")

        return "\n".join(res)

    def as_dict(self):
        """
        Represent an action "as a" dictionnary. This dictionnary is usefull to further inspect on which elements
        the actions had an impact. It is not recommended to use it as a way to serialize actions.

        The following keys (all optional) are present in the results:

          * `load_p`: if the action modifies the active loads.
          * `load_q`: if the action modifies the reactive loads.
          * `prod_p`: if the action modifies the active productions of generators.
          * `prod_v`: if the action modifies the voltage setpoint of generators.
          * `set_line_status` if the action tries to **set** the status of some powerlines. If present, this is a
            a dictionnary with keys:

              * `nb_connected`: number of powerlines that are reconnected
              * `nb_disconnected`: number of powerlines that are disconnected
              * `connected_id`: the id of the powerlines reconnected
              * `disconnected_id`: the ids of the powerlines disconnected

          * `change_line_status`: if the action tries to **change** the status of some powelrines. If present, this
            is a dictionnary with keys:

              * `nb_changed`: number of powerlines having their status changed
              * `changed_id`: the ids of the powerlines that are changed

          * `change_bus_vect`: if the action tries to **change** the topology of some substations. If present, this
            is a dictionnary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: ids of the substations impacted by the action
              * `change_bus_vect`: details the objects that are modified. It is itself a dictionnary that represents for
                each impacted substations (keys) the modification of the objects connected to it.

          * `set_bus_vect`: if the action tries to **set** the topology of some substations. If present, this is a
            dictionnary with keys:

              * `nb_modif_subs`: number of substations impacted by the action
              * `modif_subs_id`: the ids of the substations impacted by the action
              * `set_bus_vect`: details the objects that are modified. It is also a dictionnary that representes for
                each impacted substations (keys) how the elements connected to it are impacted (their "new" bus)

        Returns
        -------
        res: ``dict``
            The action represented as a dictionnary. See above for a description of it.
        """
        res = {}

        # saving the injections
        for k in ["load_p", "prod_p", "load_q", "prod_q"]:
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
                if k !=  0:
                    obj_id, objt_type, substation_id = self._obj_caract_from_topo_id(id_)
                    sub_id = "{}".format(substation_id)
                    if not sub_id in res["set_bus_vect"]:
                        res["set_bus_vect"][sub_id] = {}
                    res["set_bus_vect"][sub_id]["{}".format(obj_id)] = {"type": objt_type, "new_bus": k}
                    all_subs.add(sub_id)

            res["set_bus_vect"]["nb_modif_subs"] = len(all_subs)
            res["set_bus_vect"]["modif_subs_id"] = sorted(all_subs)

        return res

    def effect_on(self, _sentinel=None, load_id=None, gen_id=None, line_id=None, substation_id=None):
        """
        Return the effect of this action on a give unique load, generator unit, powerline of substation.
        Only one of load, gen, line or substation should be filled.

        The querry of these objects can only be done by id here (ie by giving the integer of the object in the backed).
        The :class:`HelperAction` has some utilities to access them by name too.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        load_id: ``int``
            ID of the load we want to inspect

        gen_id: ``int``
            ID of the generator we want to inspect

        line_id: ``int``
            ID of the powerline we want to inspect

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionnary with keys and value depending on which object needs to be inspected:

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

            - if a powerline is inspected then the keys are:

                - "change_bus_or" : whether or not the origin end will be moved from one bus to another
                - "change_bus_ex" : whether or not the extremity end will be moved from one bus to another
                - "set_bus_or" : the new bus where the origin will be moved
                - "set_bus_ex" : the new bus where the extremity will be moved
                - "set_line_status" : the new status of the power line
                - "change_line_status" : whether or not to switch the status of the powerline

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "change_bus"
                - "set_bus"

        NB the difference between "set_bus" and "change_bus" is the following:

            - If "set_bus" is 1, then the object (load, generator or powerline) will be moved to bus 1 of the substation
              to which it is connected. If it is already to bus 1 nothing will be done. If it's on another bus it will
              connect it to bus 1. It's it's disconnected, it will reconnect it and connect it to bus 1.
            - If "change_bus" is True, then object will be moved from one bus to another. If the object where on bus 1
              then it will be moved on bus 2, and if it were on bus 2, it will be moved on bus 1. If the object were
              disconnected, then it will be connected on the affected bus.

        Raises
        ------
        Grid2OpException
            If _sentinel is modified, or if None of the arguments are set or alternatively if 2 or more of the
            _parameters are being set.

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
            my_id = self._load_pos_topo_vect[load_id]
            res["change_bus"] = self._change_bus_vect[my_id]
            res["set_bus"] = self._set_topo_vect[my_id]
        elif gen_id is not None:
            if line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inpsect the effect of an action on one single element")
            res = {"new_p": np.NaN, "new_v": np.NaN, "set_topology": 0.}
            if "prod_p" in self._dict_inj:
                res["new_p"] = self._dict_inj["prod_p"][gen_id]
            if "prod_v" in self._dict_inj:
                res["new_v"] = self._dict_inj["prod_v"][gen_id]
            my_id = self._gen_pos_topo_vect[gen_id]
            res["change_bus"] = self._change_bus_vect[my_id]
            res["set_bus"] = self._set_topo_vect[my_id]
        elif line_id is not None:
            if substation_id is not None:
                raise Grid2OpException("You can only the inpsect the effect of an action on one single element")
            res = {}
            # origin topology
            my_id = self._lines_or_pos_topo_vect[line_id]
            res["change_bus_or"] = self._change_bus_vect[my_id]
            res["set_bus_or"] = self._set_topo_vect[my_id]
            # extremity topology
            my_id = self._lines_ex_pos_topo_vect[line_id]
            res["change_bus_ex"] = self._change_bus_vect[my_id]
            res["set_bus_ex"] = self._set_topo_vect[my_id]
            # status
            res["set_line_status"] = self._set_line_status[line_id]
            res["change_line_status"] = self._switch_line_status[line_id]
        else:
            res = {}
            beg_ = int(np.sum(self._subs_info[:substation_id]))
            end_ = int(beg_ + self._subs_info[substation_id])
            res["change_bus"] = self._change_bus_vect[beg_:end_]
            res["set_bus"] = self._set_topo_vect[beg_:end_]

        return res


class TopologyAction(Action):
    """
    This class is here to model only Topological actions.
    It will throw an "AmbiguousAction" error it someone attempt to change injections in any ways.

    It has the same attributes as its base class :class:`Action`.

    It is also here to show an example on how to implement a valid class deriving from :class:`Action`.
    """
    def __init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.
        """
        Action.__init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys if k != "injection"])

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionnary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is always empty

        set_line_status: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_line_status`

        switch_line_status: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._switch_line_status`

        set_topo_vect: :class:`numpy.array`, dtype:int
            This array is :attr:`Action._set_topo_vect`

        change_bus_vect: :class:`numpy.array`, dtype:bool
            This array is :attr:`Action._change_bus_vect`
        """
        if self._dict_inj:
            raise AmbiguousAction("You asked to modify the injection with an action of class \"TopologyAction\".")
        self._check_for_ambiguity()
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect

    def update(self, dict_):
        """
        As its original implementation, this method allows to modify the way a dictionnary can be mapped to a valid
        :class:`Action`.

        It has only minor modifications compared to the original :func:`Action.update` implementation, most notably, it
        doesn't update the :attr:`Action._dict_inj`. It raises a warning if attempting to change them.

        Parameters
        ----------
        dict_: :class:`dict`
            See the help of :func:`Action.update` for a detailed explanation. **NB** all the explanations concerning the
            "injection" part are irrelevant for this subclass.

        Returns
        -------
        self: :class:`TopologyAction`
            Return object itself thus allowing mutiple call to "update" to be chained.
        """

        self.as_vect = None
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
            self._digest_hazards(dict_)
            self._digest_maintenance(dict_)
            self._digest_change_status(dict_)
        return self

    def size(self):
        """
        Compare to the base class, this action has a shorter size, as all information about injections are ignored.
        Returns
        -------
        size: ``int``
            The size of :class:`TopologyAction` converted to an array.
        """
        return 2 * self._n_lines + 2 * self._dim_topo

    def to_vect(self):
        """
        See :func:`Action.to_vect` for a detailed description of this method.

        This method has the same behaviour as its base class, except it doesn't require any information about the
        injections to be sent, thus being more efficient from a memory footprint perspective.

        Returns
        -------
        as_vect: :class:`numpy.array`, dtype:float
            The instance of this action converted to a vector.
        """
        if self.as_vect is None:
            self.as_vect = np.concatenate((
                self._set_line_status.flatten().astype(np.float),
                self._switch_line_status.flatten().astype(np.float),
                self._set_topo_vect.flatten().astype(np.float),
                self._change_bus_vect.flatten().astype(np.float)
                              ))

            if self.as_vect.shape[0] != self.size():
                raise AmbiguousAction("Action has not the proper shape.")

        return self.as_vect

    def from_vect(self, vect):
        """
        See :func:`Action.from_vect` for a detailed description of this method.

        Nothing more is made except the initial vector is smaller.

        Parameters
        ----------
        vect: :class:`numpy.array`, dtype:float
            A vector reprenseting an instance of :class:`.`

        Returns
        -------

        """
        self.reset()
        # pdb.set_trace()
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while loading a \"TopologyAction\" from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))
        prev_ = 0
        next_ = self._n_lines

        self._set_line_status = vect[prev_:next_]; prev_ += self._n_lines; next_ += self._n_lines
        self._set_line_status = self._set_line_status.astype(np.int)
        self._switch_line_status = vect[prev_:next_]; prev_ += self._n_lines; next_ += self._dim_topo
        self._switch_line_status = self._switch_line_status.astype(np.bool)
        self._set_topo_vect = vect[prev_:next_]; prev_ += self._dim_topo; next_ += self._dim_topo
        self._set_topo_vect = self._set_topo_vect.astype(np.int)
        self._change_bus_vect = vect[prev_:]; prev_ += self._dim_topo
        self._change_bus_vect = self._change_bus_vect.astype(np.bool)

        self._check_for_ambiguity()

    def sample(self):
        """
        TODO
        Returns
        -------

        """
        self.reset()
        # TODO code the sampling now
        # TODO test it !!!
        return self


class PowerLineSet(Action):
    """
    This class is here to model only a subpart of Topological actions, the one consisting in topological switching.
    It will throw an "AmbiguousAction" error it someone attempt to change injections in any ways.

    It has the same attributes as its base class :class:`Action`.

    It is also here to show an example on how to implement a valid class deriving from :class:`Action`.

    **NB** This class doesn't allow to connect object to other buses than their original bus. In this case,
    reconnecting a powerline cannot be considered "ambiguous". We have to
    """
    def __init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect):
        """
        See the definition of :func:`Action.__init__` and of :class:`Action` for more information. Nothing more is done
        in this constructor.
        """
        Action.__init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect)

        # the injection keys is not authorized, meaning it will send a warning is someone try to implement some
        # modification injection.
        self.authorized_keys = set([k for k in self.authorized_keys
                                    if k != "injection" and k != "set_bus" and \
                                    k != "change_bus" and k != "change_line_status"])

    def __call__(self):
        """
        Compare to the ancestor :func:`Action.__call__` this type of Action doesn't allow to change the injections.
        The only difference is in the returned value *dict_injection* that is always an empty dictionnary.

        Returns
        -------
        dict_injection: :class:`dict`
            This dictionnary is always empty

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
        return {}, self._set_line_status, self._switch_line_status, self._set_topo_vect, self._change_bus_vect

    def update(self, dict_):
        """
        As its original implementation, this method allows to modify the way a dictionnary can be mapped to a valid
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
            Return object itself thus allowing mutiple call to "update" to be chained.
        """

        self.as_vect = None
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

    def size(self):
        """
        Compare to the base class, this action has a shorter size, as all information about injections are ignored.
        Returns
        -------
        size: ``int``
            The size of :class:`PowerLineSet` converted to an array.
        """
        return self._n_lines

    def to_vect(self):
        """
        See :func:`Action.to_vect` for a detailed description of this method.

        This method has the same behaviour as its base class, except it doesn't require any information about the
        injections to be sent, thus being more efficient from a memory footprint perspective.

        Returns
        -------
        as_vect: :class:`numpy.array`, dtype:float
            The instance of this action converted to a vector.
        """
        if self.as_vect is None:
            self.as_vect = self._set_line_status.flatten().astype(np.float)
            if self.as_vect.shape[0] != self.size():
                raise AmbiguousAction("PowerLineSwitch has not the proper shape.")

        return self.as_vect

    def from_vect(self, vect):
        """
        See :func:`Action.from_vect` for a detailed description of this method.

        Nothing more is made except the initial vector is (much) smaller.

        Parameters
        ----------
        vect: :class:`numpy.array`, dtype:float
            A vector reprenseting an instance of :class:`.`

        Returns
        -------

        """
        self.reset()
        # pdb.set_trace()
        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while loading a \"TopologyAction\" from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))
        prev_ = 0
        next_ = self._n_lines

        self._set_line_status = vect[prev_:next_]
        self._set_line_status = self._set_line_status.astype(np.int)
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
            self._set_topo_vect[self._lines_ex_pos_topo_vect[sel_]] = 1
            self._set_topo_vect[self._lines_or_pos_topo_vect[sel_]] = 1

    def sample(self):
        """
        Sample a PowerlineSwitch Action.

        By default, this sampling will act on one random powerline, and it will either
        disconnect it or reconnect it each with equal probability.

        Returns
        -------
        res: :class:`PowerLineSwitch`
            The sampled action
        """
        self.reset()
        i = np.random.randint(0, self.size())  # the powerline on which we can act
        val = 2*np.random.randint(0, 2) - 1  # the action: +1 reconnect it, -1 disconnect it
        self._set_line_status[i] = val
        if val == 1:
            self._set_topo_vect[self._lines_ex_pos_topo_vect[i]] = 1
            self._set_topo_vect[self._lines_or_pos_topo_vect[i]] = 1
        return self


# TODO have something that output a dict like "i want to change this element", with its name accessible here
class HelperAction:
    """
    :class:`HelperAction` should be instanciated by an :class:`Environment` with its _parameters coming from a properly
    set up :class:`Backend` (ie a Backend instance with a loaded powergrid. See :func:`grid2op.Backend.load_grid` for
    more information).

    It will allow, thanks to its :func:`HelperAction.__call__` method to create valid :class:`Action`. It is the
    preferred way to create object of class :class:`Action` in this package.

    On the contrary to the :class:`Action`, it is NOT recommended to overload this helper. If more flexibility is
    needed on the type of :class:`Action` created, it is recommended to pass a different "*actionClass*" argument
    when it's built. Note that it's mandatory that the class used in the "*actionClass*" argument derived from the
    :class:`Action`.

    Attributes
    ----------

    n_lines: :class:`int`
        number of powerline in the _grid

    n_gen: :class:`int`
        number of generators in the _grid

    n_load: :class:`int`
        number of loads in the powergrid

    subs_info: :class:`numpy.array`, dtype:int
        for each substation, gives the number of elements connected to it

    load_to_subid: :class:`numpy.array`, dtype:int
        for each load, gives the id the substation to which it is connected

    gen_to_subid: :class:`numpy.array`, dtype:int
        for each generator, gives the id the substation to which it is connected

    lines_or_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "origin" end is connected

    lines_ex_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "extremity" end is connected

    load_to_sub_pos: :class:`numpy.array`, dtype:int
        The topology if of the subsation *i* is given by a vector, say *sub_topo_vect* of size
        :attr:`HelperAction.subs_info`\[i\]. For a given load of id *l*, :attr:`HelperAction._load_to_sub_pos`\[l\] is the index
        of the load *l* in the vector *sub_topo_vect*. This means that, if
        *sub_topo_vect\[ action._load_to_sub_pos\[l\] \]=2*
        then load of id *l* is connected to the second bus of the substation.

    gen_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_to_sub_pos` but for generators.

    lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_to_sub_pos`  but for "origin" end of powerlines.

    lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_to_sub_pos` but for "extremity" end of powerlines.

    load_pos_topo_vect: :class:`numpy.array`, dtype:int
        It has a similar role as :attr:`HelperAction._load_to_sub_pos` but it gives the position in the vector representing
        the whole topology. More concretely, if the complete topology of the powergrid is represented here by a vector
        *full_topo_vect* resulting of the concatenation of the topology vector for each substation
        (see :attr:`Backend._load_to_sub_pos`for more information). For a load of id *l* in the powergrid,
        :attr:`HelperAction._load_pos_topo_vect`\[l\] gives the index, in this *full_topo_vect* that concerns load *l*.
        More formally, if *_topo_vect\[ backend._load_pos_topo_vect\[l\] \]=2* then load of id l is connected to the
        second bus of the substation.

    gen_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_pos_topo_vect` but for generators.

    lines_or_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_pos_topo_vect` but for "origin" end of powerlines.

    lines_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`HelperAction._load_pos_topo_vect` but for "extremity" end of powerlines.

    name_load: :class:`numpy.array`, dtype:str
        ordered name of the loads in the helper. This is mainly use to make sure the "chronics" are used properly.

    name_prod: :class:`numpy.array`, dtype:str
        ordered name of the productions in the helper. This is mainly use to make sure the "chronics" are used properly.

    name_line: :class:`numpy.array`, dtype:str
        ordered name of the productions in the helper. This is mainly use to make sure the "chronics" are used properly.

    template_act: :class:`Action`
        An instance of the "*actionClass*" provided used to provide higher level utilities, such as the size of the
        action (see :func:`Action.size`) or to sample a new Action (see :func:`Action.sample`)

    game_rules: :class:`grid2op.GameRules.GameRules`
        Class specifying the rules of the game, used to check the legality of the actions.

    n: ``int``
        Size of the action space
    """
    def __init__(self, name_prod, name_load, name_line, subs_info,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 game_rules, actionClass=Action):
        """
        All _parameters (name_prod, name_load, name_line, subs_info, etc.) are used to fill the attributes having the
        same name. See :class:`HelperAction` for more information.

        Parameters
        ----------
        actionClass: ``type``
            Note that this parameter expected a class, and not an object of the class. It is used to return the
            appropriate action type.

        game_rules: :class:`grid2op.GameRules.GameRules`
            Class specifying the rules of the game, used to check the legality of the actions.

        """

        self.name_prod = name_prod
        self.name_load = name_load
        self.name_line = name_line

        self.n_gen = len(name_prod)
        self.n_load = len(name_load)
        self.n_lines = len(name_line)

        self.subs_info = subs_info
        self.dim_topo = np.sum(subs_info)
        self.actionClass = actionClass

        # to which substation is connected each element
        self.load_to_subid = load_to_subid
        self.gen_to_subid = gen_to_subid
        self.lines_or_to_subid = lines_or_to_subid
        self.lines_ex_to_subid = lines_ex_to_subid
        # which index has this element in the substation vector
        self.load_to_sub_pos = load_to_sub_pos
        self.gen_to_sub_pos = gen_to_sub_pos
        self.lines_or_to_sub_pos = lines_or_to_sub_pos
        self.lines_ex_to_sub_pos = lines_ex_to_sub_pos
        # which index has this element in the topology vector
        self.load_pos_topo_vect = load_pos_topo_vect
        self.gen_pos_topo_vect = gen_pos_topo_vect
        self.lines_or_pos_topo_vect = lines_or_pos_topo_vect
        self.lines_ex_pos_topo_vect = lines_ex_pos_topo_vect

        self.template_act = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)

        self.legal_action = game_rules.legal_action

        self.n = self.template_act.size()

    def __call__(self, dict_=None, check_legal=False, env=None):
        """
        This utility allows you to build a valid action, with the proper sizes if you provide it with a valid
        dictionnary.

        More information about this dictionnary can be found in the :func:`Action.__call__` help. This dictionnary
        is not changed in this method.

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
            Properly instanciated.

        """

        res = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)
        # update the action
        res.update(dict_)
        if check_legal:
            if not self._is_legal(res, env):
                raise IllegalAction("Impossible to perform action {}".format(res))

        return res

    def sample(self):
        """
        A utility used to sample action.
        Returns
        -------
        res: :class:`Action`
            A random action sampled from the :attr:`HelperAction.actionClass`
        """
        res = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)
        res.sample()
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

    def change_bus(self, name_element, extremity=None, substation=None, type_element=None, previous_action=None):
        """
        Utilities to change the bus of a single element if you give its name. **NB** Chaning a bus has the effect to
        assign the object to bus 1 if it was before that connected to bus 2, and to assign it to bus 2 if it was
        connected to bus 1. It should not be mixed up with :func:`HelperAction.set_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (inplace) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus
        extremity: ``str``
            "or" or "ex" for origin or extremity, ignored if element is not a powerline.
        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise the method will search it.
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
        AmbiguousAction
            If *previous_action* has not the same type as :attr:`HelperAction.actionClass`.
        """
        if previous_action is None:
            res = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)
        else:
            if not isinstance(previous_action, self.actionClass):
                raise AmbiguousAction("The action to update using `HelperAction` is of type \"{}\" which is not the type of action handled by this helper (\"{}\")".format(type(previous_action), self.actionClass))
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation, type_element, res)
        dict_["change_bus"][to_sub_pos[my_id]] = True
        res.update({"change_bus": {"substations_id": [(my_sub_id, dict_["change_bus"])]}})
        # res.update(dict_)
        return res

    def _extract_database_powerline(self, extremity):
        if extremity[:2] == "or":
            to_subid = self.lines_or_to_subid
            to_sub_pos = self.lines_or_to_sub_pos
            to_name = self.name_line
        elif extremity[:2] == "ex":
            to_subid = self.lines_ex_to_subid
            to_sub_pos = self.lines_ex_to_sub_pos
            to_name = self.name_line
        elif extremity is None:
            raise Grid2OpException("It is mandatory to know on which ends you want to change the bus of the powerline")
        else:
            raise Grid2OpException("unknown extremity specifier \"{}\". Extremity should be \"or\" or \"ex\"".format(extremity))
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
            to_name = self.name_prod
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
            elif name_element in self.name_prod:
                to_subid = self.gen_to_subid
                to_sub_pos = self.gen_to_sub_pos
                to_name = self.name_prod
            elif name_element in self.name_line:
                to_subid, to_sub_pos, to_name = self._extract_database_powerline(extremity)
            else:
                AmbiguousAction(
                    "Element \"{}\" not found in the powergrid".format(
                        name_element))
        else:
            raise AmbiguousAction("unknown type_element specifier \"{}\". type_element should be \"line\" or \"load\" or \"gen\"".format(extremity))

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
        = 1) it will stay on bus 1. If it was on bus 2 (and you still assign it to bus 1) it will be moved to bus
        1. It should not be mixed up with :func:`HelperAction.change_bus`.

        If the parameter "*previous_action*" is not ``None``, then the action given to it is updated (inplace) and
        returned.

        Parameters
        ----------
        name_element: ``str``
            The name of the element you want to change the bus

        new_bus: ``int``
            Id of the new bus to connect the object to.

        extremity: ``str``
            "or" or "ext" for origin or extremity, ignored if element is not a powerline.

        substation: ``int``, optional
            Its substation ID, if you know it will increase the performance. Otherwise the method will search it.

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
            res = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)
        else:
            res = previous_action

        dict_, to_sub_pos, my_id, my_sub_id = self._extract_dict_action(name_element, extremity, substation, type_element, res)
        dict_["set_bus"][to_sub_pos[my_id]] = new_bus
        res.update({"set_bus": {"substations_id": [(my_sub_id, dict_["set_bus"])]}})
        return res

    def reconnect_powerline(self, l_id, bus_or, bus_ex, previous_action=None):
        """
        Build the valid not ambiguous action consisting in reconnecting a powerline.

        Parameters
        ----------
        l_id: `int`
            the powerline id to be reconnected
        bus_or: `int`
            the bus to which connect the origin end of the powerline
        bus_ex: `int`
            the bus to which connect the extremity end the powerline

        Returns
        -------

        """
        if previous_action is None:
            res = self.actionClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                               load_to_subid=self.load_to_subid,
                               gen_to_subid=self.gen_to_subid,
                               lines_or_to_subid=self.lines_or_to_subid,
                               lines_ex_to_subid=self.lines_ex_to_subid,
                               load_to_sub_pos=self.load_to_sub_pos,
                               gen_to_sub_pos=self.gen_to_sub_pos,
                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                               load_pos_topo_vect=self.load_pos_topo_vect,
                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect)
        else:
            res = previous_action
        res.update({"set_line_status": [(l_id, 1)],
                    "set_bus": {"lines_or_id": [(l_id, bus_or)], "lines_ex_id": [(l_id, bus_ex)]}
                    })
        return res

    def size(self):
        """
        The size of any action converted to vector.
        Returns
        -------
        n: ``int``
            The size of the action space.
        """
        return self.n

    def get_set_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "set_status" keyword if building an :class:`Action`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.int
            A vector that doesn't affect the grid, but can be used in "set_line_status"

        """
        return self.template_act.get_set_line_status_vect()

    def get_change_line_status_vect(self):
        """
        Computes and return a vector that can be used in the "change_line_status" keyword if building an :class:`Action`

        Returns
        -------
        res: :class:`numpy.array`, dtype:np.bool
            A vector that doesn't affect the grid, but can be used in "change_line_status"

        """
        return self.template_act.get_change_line_status_vect()

    def from_vect(self, act):
        """
        Convert an action, represented as a vector to a valid :class:`Action` instance

        Parameters
        ----------
        act: ``numpy.ndarray``
            A action represented as a vector

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The corresponding action as an :class:`Action` instance

        """
        res = copy.deepcopy(self.template_act)
        res.from_vect(act)
        return res
