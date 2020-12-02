.. _name_load: ./space.html#grid2op.Space.GridObjects.name_load
.. _name_gen: ./space.html#grid2op.Space.GridObjects.name_gen
.. _name_line: ./space.html#grid2op.Space.GridObjects.name_line
.. _name_sub: ./space.html#grid2op.Space.GridObjects.name_sub
.. _n_line: ./space.html#grid2op.Space.GridObjects.n_line
.. _n_gen: ./space.html#grid2op.Space.GridObjects.n_gen
.. _n_load: ./space.html#grid2op.Space.GridObjects.n_load
.. _n_sub: ./space.html#grid2op.Space.GridObjects.n_sub
.. _sub_info: ./space.html#grid2op.Space.GridObjects.sub_info
.. _dim_topo: ./space.html#grid2op.Space.GridObjects.dim_topo
.. _load_to_subid: ./space.html#grid2op.Space.GridObjects.load_to_subid
.. _gen_to_subid: ./space.html#grid2op.Space.GridObjects.gen_to_subid
.. _line_or_to_subid: ./space.html#grid2op.Space.GridObjects.line_or_to_subid
.. _line_ex_to_subid: ./space.html#grid2op.Space.GridObjects.line_ex_to_subid
.. _load_to_sub_pos: ./space.html#grid2op.Space.GridObjects.load_to_sub_pos
.. _gen_to_sub_pos: ./space.html#grid2op.Space.GridObjects.gen_to_sub_pos
.. _line_or_to_sub_pos: ./space.html#grid2op.Space.GridObjects.line_or_to_sub_pos
.. _line_ex_to_sub_pos: ./space.html#grid2op.Space.GridObjects.line_ex_to_sub_pos
.. _load_pos_topo_vect: ./space.html#grid2op.Space.GridObjects.load_pos_topo_vect
.. _gen_pos_topo_vect: ./space.html#grid2op.Space.GridObjects.gen_pos_topo_vect
.. _line_or_pos_topo_vect: ./space.html#grid2op.Space.GridObjects.line_or_pos_topo_vect
.. _line_ex_pos_topo_vect: ./space.html#grid2op.Space.GridObjects.line_ex_pos_topo_vect

.. |5subs_grid_layout| image:: ./img/5subs_grid_layout.jpg
.. |5subs_grid_1_sub| image:: ./img/5subs_grid_1_sub.jpg
.. |5subs_grid_2_loads| image:: ./img/5subs_grid_2_loads.jpg
.. |5subs_grid_3_gens| image:: ./img/5subs_grid_3_gens.jpg
.. |5subs_grid_4_lines| image:: ./img/5subs_grid_4_lines.jpg
.. |5subs_grid_5_obj_in_sub| image:: ./img/5subs_grid_5_obj_in_sub.jpg
.. |5subs_grid_layout_with_repr| image:: ./img/5subs_grid_layout_with_repr.jpg
.. |5subs_grid_n_el| image:: ./img/5subs_grid_n_el.jpg
.. |5subs_grid_5_sub_i| image:: ./img/5subs_grid_5_sub_i.jpg
.. |5subs_grid_load_to_subid| image:: ./img/5subs_grid_load_to_subid.jpg
.. |5subs_grid_el_to_subid| image:: ./img/5subs_grid_el_to_subid.jpg
.. |5subs_grid_sub0| image:: ./img/5subs_grid_sub0.jpg
.. |5subs_grid_sub0_final| image:: ./img/5subs_grid_sub0_final.jpg
.. |5subs_grid_sub1_final| image:: ./img/5subs_grid_sub1_final.jpg

.. _create-backend-module:

Creating a new backend
===================================

Objectives
-----------

.. warning:: Backends are internal to grid2op.

    This page details how to create a backend from an available
    solver (a "things" that is able to compute flows and voltages). This is an advanced usage.

    You will also find in this file the complete description on how the "powergrid" is represented in grid2op.

    Backend is an abstraction that represents the physical system (the powergrid). In theory every powerflow can be
    used as a backend. For now we only provide a Backend that uses `Pandapower <http://www.pandapower.org/>`_ and
    a port in c++ to a subset of pandapower called `LightSim2Grid <https://github.com/BDonnot/lightsim2grid>`_ .

    Both can serve as example if you want to code a new backend.

To implement completely a backend, you should implement all the abstract function defined here :ref:`backend-module`.
This file is an overview of what is needed and aims at contextualizing these requirements to make them clearer.

This section you have already a "code" that is able to compute some powerflow given a file stored on
a hard typically representing a powergrid. If you have that, you can probably use it to implement
a grid2op backend and benefit from the whole grid2op ecosystem (code once a backend, reuse your "powerflow"
everywhere). This includes, but is not limited to:

- Save "logs" of your experiment in a standard format (with the runner) to be reused and analyzed graphically
  with grid2viz
- Save "logs" of your experiments and compare the results with "reference" solver available
- Act on your powergrid with a unified and "somewhat standard" fashion thanks to grid2op actions
- Reuse agents that other people have trained in the context of L2RPN competitions
- Train new grid controlers using the grid2op gym_compat module
- etc.

Main methods to implement
--------------------------
Typically, a backend has a internal "modeling" / "representation" of the powergrid
stored in the attribute `self._grid` that can be anything.

There are 4 **__main__** types of method you need to implement if you want to use a custom powerflow
(*eg* from a physical solver, from a neural network, or any other methods):

- :func:`grid2op.Backend.Backend.load_grid` where the environment informs the instance of your backend of where
  the grid file is located. It is expected that this function defines all the attributes listed in
  :class:`grid2op.Space.GridObjects` (more information about these attributes are given in the :ref:`grid-description`
  section of this file. It should not return anything. Its main goal is to "inform" grid2op about
  relevant information of the current powergrid and to initialize the `_grid` attribute of the backend.
- :func:`grid2op.Backend.Backend.apply_action`: that modifies the internal state of the "Backend" you create
  properly (given the action taken by the agents, or the modifications of the data, or the emulation of some
  "automaton" of the environment etc.). More detail on how to "understand" a "BackendAction" is given in the
  :ref:`backend-action-create-backend` section of this document. This function should not return anything. Its
  main goal is to allow the modification of the underlying powergrid from the environment.
- :func:`grid2op.Backend.Backend.runpf` is called by the environment when a new "simulation" should be carried
  out. It should return ``True`` if it has converged, or ``False`` otherwise. In case of non convergence (this
  function returns ``False``),
  no flows can be inspected on the internal grid and the "environment" will interpret it as a "game over".
- the "readers" functions (*eg.* :func:`grid2op.Backend.Backend.get_topo_vect`,
  :func:`grid2op.Backend.Backend.generators_info`, :func:`grid2op.Backend.Backend.loads_info`,
  :func:`grid2op.Backend.Backend.lines_or_info`, :func:`grid2op.Backend.Backend.lines_ex_info` or
  :func:`grid2op.Backend.Backend.shunt_info`) that allows to "export" data from the internal backend representation
  to a format the environment understands (*ie* vectors). You can consult the section
  :ref:`vector-orders-create-backend` of this document for more information. The main goal of these "getters" is
  to export some internal value of the backend in a "grid2op compliant format".


.. _grid-description:

Grid description
------------------
In this section we explicit what attributes need to be implemented to have a valid backend instance. We focus on
the attribute of the `Backend` you have to set. But don't forget you also need to load a powergrid and store
it in the `_grid` attribute.

The grid2op attributes that need to be implemented in the :func:`grid2op.Backend.Backend.load_grid` function are
given in the table bellow:

=========================  ==============  ===========  =========  =========================================================
Name                       See paragraph   Type         Size       Description
=========================  ==============  ===========  =========  =========================================================
`name_load`_                               vect, str    `n_load`_  (optional) name of each load on the grid [if not set, by default it will be "load_$LoadSubID_$LoadID" for example "load_1_10" if the load with id 10 is connected to substation with id 1]
`name_gen`_                                vect, str    `n_gen`_   (optional) name of each generator on the grid [if not set, by default it will be "gen_$GenSubID_$GenID" for example "gen_2_42" if the generator with id 42 is connected to substation with id 2]
`name_line`_                               vect, str    `n_line`_  (optional) name of each powerline (and transformers !) on the grid [if not set, by default it will be "$SubOrID_SubExID_LineID" for example "1_4_57" if the powerline with id 57 has its origin end connected to substation with id 1 and its extremity end connected to substation with id 4]
`name_sub`_                                vect, str    `n_sub`_   (optional) name of each substation on the grid [if not set, by default it will be "sub_$SubID" for example "sub_41" for the substation with id 41]
`n_line`_                   :ref:`n-el`    int          NA          Number of powerline on the grid (remember, in grid2op framework a `powerline` includes both "powerlines" and "transformer")
`n_gen`_                    :ref:`n-el`    int          NA          Number of generators on the grid
`n_load`_                   :ref:`n-el`    int          NA          Number of loads on the grid
`n_sub`_                    :ref:`n-el`    int          NA          Number of substations on the grid
`sub_info`_                 :ref:`sub-i`   vect, int    `n_sub`_    For each substation, it gives the number of elements connected to it ("elements" here denotes: powerline - and transformer- ends, load or generator)
`dim_topo`_                 :ref:`sub-i`   int          NA          Total number of elements on the grid ("elements" here denotes: powerline - and transformer- ends, load or generator)
`load_to_subid`_            :ref:`subid`   vect, int    `n_load`_   For each load, it gives the substation id to which it is connected
`gen_to_subid`_             :ref:`subid`   vect, int    `n_gen`_    For each generator, it gives the substation id to which it is connected
`line_or_to_subid`_         :ref:`subid`   vect, int    `n_line`_   For each powerline, it gives the substation id to which its **origin** end is connected
`line_ex_to_subid`_         :ref:`subid`   vect, int    `n_line`_   For each powerline, it gives the substation id to which its **extremity** end is connected
`load_to_sub_pos`_          :ref:`subpo`   vect, int    `n_load`_   See the description for more information ("a picture often speaks a thousand words")
`gen_to_sub_pos`_           :ref:`subpo`   vect, int    `n_gen`_    See the description for more information ("a picture often speaks a thousand words")
`line_or_to_sub_pos`_       :ref:`subpo`   vect, int    `n_line`_   See the description for more information ("a picture often speaks a thousand words")
`line_ex_to_sub_pos`_       :ref:`subpo`   vect, int    `n_line`_   See the description for more information ("a picture often speaks a thousand words")
`load_pos_topo_vect`_       :ref:`subtv`   vect, int    `n_load`_   Automatically set with a call to `self._compute_pos_big_topo`
`gen_pos_topo_vect`_        :ref:`subtv`   vect, int    `n_gen`_    Automatically set with a call to `self._compute_pos_big_topo`
`line_or_pos_topo_vect`_    :ref:`subtv`   vect, int    `n_line`_   Automatically set with a call to `self._compute_pos_big_topo`
`line_ex_pos_topo_vect`_    :ref:`subtv`   vect, int    `n_line`_   Automatically set with a call to `self._compute_pos_big_topo`
=========================  ==============  ===========  =========  =========================================================


Example on how to set them
+++++++++++++++++++++++++++
Some concrete example on how to create a backend are given in the :class:`grid2op.Backend.PandaPowerBackend`
(for the default Backend) and in the "lightsim2grid" backend (available at
`https://github.com/BDonnot/lightsim2grid <https://github.com/BDonnot/lightsim2grid>`_ ). Feel free to consult
any of these codes for more information.

In this example, we detail what is needed to create a backend and how to set the required attributes.

We explain step by step how to proceed with this powergid:

|5subs_grid_layout|

.. _pre-req-backend:

Prerequisite: Order and label everything
*****************************************
The first step is to give names and order to every object on the loaded grid.


For example, you can first assign order to substations this way:

|5subs_grid_1_sub|

.. warning:: To be consistent with python ecosystem, index are 0 based. So the first element should have id 0 (and not 1)

Then you decide an ordering of the loads:

|5subs_grid_2_loads|

Then the generators:

|5subs_grid_3_gens|

And then you deal with the powerlines. Which is a bit more "complex" as you need also to "orient" each powerline
which is to define an "origin" end and and "extremity" end. This result in a possible ordering this way:

|5subs_grid_4_lines|

Finally you also need to come up with a way of assing to each "element" an order in the substation. This is an
extremely complex way to say you have to do this:

|5subs_grid_5_obj_in_sub|

Note the number for each element in the substation.

In this example, for substaion with id 0 (bottom left) you decided
that the powerline with id 0 (connected at this substation at its origin end) will be the "first object of this
substation". Then the "Load 0" is the second object [remember index a 0 based, so the second object has id 1],
generator 0 is the third object of this substation (you can know it with the "3" near it) etc.

.. note:: Grid2op assumes that if the same files is loaded multiple times, then the same grid is defined by the
    backend. This entails that the loads are in the same order, substations are in the same order, generators are
    in the same order, powerline are in the same order (and for each powerrline is oriented the same way: same "origin"
    and same "extremity").

This powergrid will be used throughout this example. And in the next sections, we suppose that you have chosen
a way to assign all these "order".

.. note:: The order of the elements has absolutely no impact whatsoever on the solver and the state of the grid. In
    other words flows, voltages etc. do not depend on this (arbitrary) order.

    We could have chosen a different representation of this data
    (for example by identifying objects with names instead of ID in vector) but it turns out this "index based
    representation" is extremely fast as it allows manipulation of most data using the `numpy` package.

    This is a reason why grid2op is relatively fast in most cases: very little time is taken to map objects to
    there properties.

Final result
******************
For the most impatient readers, the final representation is :

|5subs_grid_layout_with_repr|

In the next paragraphs we detail step by step why this is this way.

.. _n-el:

Number of elements (n_line, n_load, n_gen, n_sub)
**************************************************
Nothing much to say here, you count each object and assign the right val to the attributes. This gives:

|5subs_grid_n_el|

For example, `n_line` is 8 because there are 8 lines on the grid, labeled from 0 to 7.

.. _sub-i:

Substation information (sub_info, dim_topo)
********************************************
For these attributes too, there is nothing really surprising.

For each component of `sub_info` you inform grid2op of the number of elements connected to it. And then you sum
up each of these elements in the `dim_topo` attributes.

|5subs_grid_5_sub_i|

.. note:: Only the loads, line ends ("origin" or "extremity") and generators are counted as "elements".

.. _subid:

Substation id (\*_to_subid)
***************************

The attribute explained in this section are `load_to_subid`, `gen_to_subid`, `line_or_to_subid` and `line_ex_to_subid`.

Again, for each of these vector, you specify to which substation the objects are connected. For example, for the
loads this gives:

|5subs_grid_load_to_subid|

Indeed, the load with id 0 is connected to substation with id 0, load with id 1 is connected to substation with id 3
and load with id 2 is connected to substation with id 4.

For the other attributes, you follow the same pattern:

|5subs_grid_el_to_subid|

.. _subpo:

Position in substation (\*_to_sub_pos)
**************************************

These are the least common (and "most complicated") attributes to set.

This values allow to uniquely identified, inside each substation. These were represented by the "small" number
near each element in the last image of the introductory paragraph :ref:`pre-req-backend`. If you have that
image in mind, it's simple: you set the number of each elements into its vector. And that is it.

If you are confused, we made a detailed example below.

First, have a look at substation 0:

|5subs_grid_sub0|

You know that, at this substation 0 there are `6` elements connected. In this example, these are:

- origin end of Line 0
- Load 0
- gen 0
- origin end of line 1
- origin end of line 2
- origin end of line 3

Given that, you can fill:

- first component of `line_or_to_sub_pos`  [origin of line 0 is connected at this substation]
- first component of `load_to_sub_pos` [Load 0 is connected at this substation]
- first component of `gen_to_sub_pos`  [gen 0 is connected at this substation]
- second component of `line_or_to_sub_pos` [origin of line 1 is connected at this substation]
- third component of `line_or_to_sub_pos`  [origin of line 2 is connected at this substation]
- fourth component of `line_or_to_sub_pos`  [origin of line 2 is connected at this substation]

These are indicated with the "??" on the figure above. (note that the `XX` cannot be set right now)

Once you know that, you just have to recall that you already give an order to each of these objects.
You defined (in a purely arbitrary manner):

- the element 0 of this substation to be "origin of line 0"
- the element 1 of this substation to be "load 0"
- the element 2 of this substation to be "gen 0"
- the element 3 of this substation to be "origin of line 3"
- the element 4 of this substation to be "origin of line 2"
- the element 5 of this substation to be "origin of line 1"

So you get:

- first component of `line_or_to_sub_pos` is 0 [because "origin end of line 0" is "element 0" of this substation]
- first component of `load_to_sub_pos` is 1 [because "load 0" is "element 1" of this substation]
- first component of `gen_to_sub_pos` is 2 [because "gen 0" is "element 2" of this substation]
- fourth component of `line_or_to_sub_pos` is 3 [because "origin end of line 3" is "element 3" of this substation]
- third component of `line_or_to_sub_pos` is 4 [because "origin end of line 2" is "element 4" of this substation]
- second component of `line_or_to_sub_pos` is 5 [because "origin end of line 1" is "element 5" of this substation]

This is showed in the figure below:

|5subs_grid_sub0_final|

Then you do the same process with substation 1 which will result in the vector showed in the following plot:

|5subs_grid_sub1_final|

When writing this, we realize it's really verbose. But really, it simply consist on assigning, at each object, a unique
ID to be able to retrieved it when querying something like "set the object of id 2 in subtation 0 to busbar 2".


Finally, you proceed in the same manner for all substation and you get:

|5subs_grid_layout_with_repr|

.. _subtv:

Position in the topology vector (\*_pos_topo_vect)
**************************************************

This information is redundant with the other vector. It can be initialized with
a call to the function :func:`grid2op.Space.GridObjects._compute_pos_big_topo` that you will need to perform after
having initialized all the other vectors as explained above (simply call `self._compute_pos_big_topo()` at the end
of your implementation of `load_grid` function)

.. _backend-action-create-backend:

BackendAction: modification
----------------------------------------------
In this section we detail step by step how to understand the specific format used by grid2op to "inform" the backend
on how to modify its internal state before computing a powerflow.

A `BackendAction` will tell the backend on what is modified among:

- the active value of each loads (see paragraph :ref:`change-inj`)
- the reactive value of each loads (see paragraph  :ref:`change-inj`)
- the amount of power produced by each generator (setpoint) (see paragraph  :ref:`change-inj`)
- the voltage "setpoint" of each generator (see paragraph  :ref:`change-inj`)
- the status (connected / disconnected) of each element (see paragraph :ref:`change-topo`)
- at which busbar each object is connected  (see paragraph :ref:`change-topo`)

.. note:: Typically the `apply_action` function is called once per `step` of the environment. The implementation of
    this function should be rather optimized for the best performance.

In this section we detail the format of these particular "actions". We assume in the following that
`backendAction` is the action you need to perform.

At the end, the `apply_action` function of the backend should look something like:

.. code-block:: python

    def apply_action(self, backendAction=None):
        if backendAction is None:
            return
        active_bus, (prod_p, prod_v, load_p, load_q), topo__, shunts__ = backendAction()

        # modify the injections [see paragraph "Modifying the injections (productions and loads)"]
        for gen_id, new_p in prod_p:
            # modify the generator with id 'gen_id' to have the new setpoint "new_p" for production
            ...  # the way you do that depends on the `internal representation of the grid`
        for gen_id, new_v in prod_v:
            # modify the generator with id 'gen_id' to have the new value "new_v" as voltage setpoint
            ...  # the way you do that depends on the `internal representation of the grid`
        for load_id, new_p in load_p:
            # modify the load with id 'load_id' to have the new value "new_p" as consumption
            ...  # the way you do that depends on the `internal representation of the grid`
        for load_id, new_p in load_p:
            # modify the load with id 'load_id' to have the new value "new_p" as consumption
            ...  # the way you do that depends on the `internal representation of the grid`

        # modify the topology [see paragraph "Modifying the topology (status and busbar)"]
        for el_id, new_bus in topo__:
            # modify the "busbar" of the object
            # the object is identified with an id corresponding to the the `\*\*_pos_topo_vect` vector
            # Please have a look at the paragraph [see paragraph "Modifying the topology (status and busbar)"]
            # for more information
            if new_bus == -1:
                # the object is disconnected in the action, disconnect it on your internal representation of the grid
                ... # the way you do that depends on the `internal representation of the grid`
            else:
                # the object is moved to either busbar 1 (in this case `new_bus` will be `1`)
                # or to busbar 2 (in this case `new_bus` will be `2`)
                ... # the way you do that depends on the `internal representation of the grid`

.. _modif-backend:

Retrieve what has been modified
++++++++++++++++++++++++++++++++

This is the first step you would need to perform to retrieve what are the modifications you need to implement
in the backend. This is achieved with:

.. code-block:: python

    active_bus, (prod_p, prod_v, load_p, load_q), topo__, shunts__ = backendAction()

And all information needed to set the state of your backend is now available. We will explain them step by step in the
following paragraphs.

.. _change-inj:

Modifying the injections (productions and loads)
+++++++++++++++++++++++++++++++++++++++++++++++++

The new setpoints for the injections are given in the vectors `(prod_p, prod_v, load_p, load_q)`  retrieved in the
above paragraph (see :ref:`modif-backend` for more information).

Each of the `prod_p`, `prod_v`, `load_p` and  `load_q` are specific types of "iterable" that stores which values
have been modified and what is the new value associated.

The first way to retrieve the modification is with a simple for loop:

.. code-block:: python

    for gen_id, new_p in prod_p:
        # modify the generator with id 'gen_id' to have the new setpoint "new_p"
        ...  # the way you do that depends on the `internal representation of the grid`

.. note::  If no changes have affected the "active production setpoint of generator" then it will not be
    "looped through": only
    generators that have been modified between the steps will be showed here. So if you need to passe the values of
    all generators (for example) you need to remember these values yourself.

Of course it works the same way with the other "iterables":

.. code-block:: python

    for gen_id, new_v in prod_v:
        # modify the generator with id 'gen_id' to have the new value "new_v" as voltage setpoint
        ...  # the way you do that depends on the `internal representation of the grid`
    for load_id, new_p in load_p:
        # modify the load with id 'load_id' to have the new value "new_p" as consumption
        ...  # the way you do that depends on the `internal representation of the grid`
    for load_id, new_p in load_p:
        # modify the load with id 'load_id' to have the new value "new_p" as consumption
        ...  # the way you do that depends on the `internal representation of the grid`


.. _change-topo:

Modifying the topology (status and busbar)
++++++++++++++++++++++++++++++++++++++++++

TODO !


.. _vector-orders-create-backend:

Vector representation of the grid information
-----------------------------------------------
TODO

.. include:: final.rst