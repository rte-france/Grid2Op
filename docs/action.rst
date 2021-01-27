.. currentmodule:: grid2op.Action
.. _action-module:

Action
===================================

Objectives
----------
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

The BaseAction class is abstract. You can implement it the way you want. If you decide to extend it, make sure
that the :class:`grid2op.Backend` class will be able to understand it. If you don't, your extension will
not affect the
underlying powergrid. Indeed a :class:`grid2op.Backend` will call the :func:`BaseAction.__call__` method
and should
understands its return type.


The :class:`BaseAction` and all its derivatives also offer some usefull inspection utilities:

  - :func:`BaseAction.__str__` prints the action in a format that gives usefull information on how it will
    affect the powergrid
  - :func:`BaseAction.effect_on` returns a dictionary that gives information about its effect.

From :class:`BaseAction` inherit in particular the :class:`PlayableAction`, the base class of all action that
players are allowed to play.

Finally, :class:`BaseAction` class define some strict behavior to follow if reimplementing them.
The correctness of each
instances of BaseAction is assessed both when calling :func:`BaseAction.update` or with a call to
:func:`BaseAction._check_for_ambiguity` performed for example by the Backend when it must implement
its effect on the
powergrid through a call to :func:`BaseAction.__call__`

Constructing an action in grid2op is made in the following manner:

.. code-block:: python

    import grid2op
    env = grid2op.make()
    dictionary_describing_the_action = {...}  # se bellow
    my_action = env.action_space(dictionary_describing_the_action)
    print(my_action)

On the above code, `dictionary_describing_the_action` should be a dictionary that describe what action
you want to perform on the grid. For more information you can consult the help of the :func:`BaseAction.update`.

To avoid extremely verbose things, as of grid2op 1.5.0, we introduced some convenience functions to allow
easier action construction. You can now do `act.load_set_bus = ...` instead of the previously way
more verbose `act.update({"set_bus": {"loads_id": ...}}`

.. _action-module-examples:

Usage Examples
--------------
In this section, we describe how to implement some action types. For further information about the
impact of the action implemented, please consult the appropriate getting_started notebook.

Set bus
++++++++
TODO

Change bus
++++++++++
TODO

Set status
+++++++++++
TODO

Change status
++++++++++++++
TODO

Redispatching
++++++++++++++
TODO

Storage power setpoint
+++++++++++++++++++++++

TODO

Getting the resulting topology after an action
+++++++++++++++++++++++++++++++++++++++++++++++
Unfortunately, it is sometimes relatively difficult to understand what will be the exact effect of a given
action on a powergrid.

This mainly caused by the fact that the modeled environment embed some complexity of a real powergrid.

To ease the process of estimating the impact of an action on a environment, tow main functions have been
developed and are available:

- `obs.simulate(act, time_step=0)` which will "apply" the action on the known state and do "as if" a step
  has been made. This is called "simulate", it is rather accurate (up to the "we don't know the future" part)
  in the sense that is does check for illegal actions, ambiguous actions, reconnect properly the powerlines
  if needed etc. and performs simulation of "cascading failures" and other things. Of course it takes
  a lot of time to carry out all these computation.
- `impact = obs + act` (since grid2op 1.5.0). On the other hand, the "+" operator of the observation is
  much faster. It can be use to rapidly estimate the state of the grid (especially the topology)
  after the application of an action for example. This is to ease the process of studying what does
  an action exactly.

The difference in computation time, for an action of type "topology set" is shown in the table below:

================================  ================================  =============  =======================
method                             env name                         backend used    time to perform (ms)
================================  ================================  =============  =======================
`obs + act`                        l2rpn_case14_sandbox             pandapower      0.21
`obs.simulate(act, time_step=0)`   l2rpn_case14_sandbox             pandapower      17.3
`obs + act`                        l2rpn_case14_sandbox             lightsim2grid   0.21
`obs.simulate(act, time_step=0)`   l2rpn_case14_sandbox             lightsim2grid   1.56
`obs + act`                        l2rpn_neurips_2020_track2_small  pandapower      0.22
`obs.simulate(act, time_step=0)`   l2rpn_neurips_2020_track2_small  pandapower      33.4
`obs + act`                        l2rpn_neurips_2020_track2_small  lightsim2grid   0.22
`obs.simulate(act, time_step=0)`   l2rpn_neurips_2020_track2_small  lightsim2grid   2.03
================================  ================================  =============  =======================

(results were obtained with grid2op version 1.5.0 on a "Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz" using
"Python 3.8.5 (default, Jul 28 2020, 12:59:40) [GCC 9.3.0] on linux" on ubuntu 20.04.1
"20.04.1-Ubuntu SMP Tue Jan 12 16:39:47 UTC 2021" using linux kernel "5.8.0-38-generic")

As you can see, the `obs + act` method is always approximately 10 times faster than the
`obs.simulate(act, time_step=0)` [of course providing much less information] and can be up
to 150 faster on larger grid (IEEE 118) using the default pandapower backend.

We can also note that, as it don't require the use of any simulation, the time to do the `obs + act` is
more or less independent of the grid size (0.21 ms for a grid counting 14 substations and
0.22ms for a grid with 118 substations) while the `obs.simulate` is not.

Now to retrieve a "graph like" object, you can :

.. code-block:: python

    # method 1
    sim_obs, *_ = obs.simulate(act)

    # method 2
    obs_add = obs + add

And refer to the section :ref:`observation_module_graph` to retrieve a graph structure from these observations.
For example:

.. code-block:: python

    bus_bus_mat = obs.bus_connectivity_matrix()


.. _Illegal-vs-Ambiguous:

Illegal vs Ambiguous
---------------------
Manipulating a powergrid is more complex than asking "pacman" to move "left" / "down" / "right" or "up". Computing
a correct action can be a tedious process.

An action can be incorrect because of two main factors:

- ``ambiguous``: this will be the case when an action is performed on 17 objects whereas the given substations counts
  only 16 of them, this will be the case when you ask to reconnect powerline 999 while there are only 20 powerlines
  on the grid etc. This is raised when the action **cannot** be understood as a correct action. Grid2op does not
  know how to interpret your action. If we take the "PacMan" game an ambiguous action would translate in moving
  "up" **and** "down" at the same time.
- ``illegal``: (see :class:`grid2op.Rules.BaseRules` and :class:`grid2op.Parameters.Parameters` for more information).
  An action can be legal or illegal depending on the rules of the game. For example, we could forbid to reconnect
  powerline 7 between time steps 123 and 159 (this would corresponds to a "maintenance" of the powerline, you can
  imagine people painting the tower for example). But that does not mean reconnecting powerline 7 is forbidden at
  other times steps. In this case we say the action is "illegal". Still my overall favorite game, in PacMan this
  would be the equivalent to moving left while there are a wall on the left.

Ambiguous or Illegal, the action will be replaced by a "do nothing" without any other incidents on the game.

Note on powerline status
------------------------
As of grid2op version 1.2.0, we attempted to clean and rationalize the API concerning the change of
powerline status (see explanatory notebook `getting_started/3_Action_GridManipulation` for more detailed
explanation.

The powerline status (connected / disconnected) can now be affected in two different ways:

- by `setting` / `changing` its status directly (using the "set_line_status" or "change_line_status" keyword).
- [NEW] by modifying the bus on any of the end (origin or extremity) of a powerline

In that later case, the behavior is:

- if the bus of a powerline end (origin or extremity) is "set" to -1 and not modified at the other and if the powerline
  was connected, it will disconnect this powerline
- if the bus of a powerline end (origin or extremity) is "set" to 1 or 2 at one end and not modified at the other and
  if the powerline was connected, it will reconnect the powerline
- if the bus of a powerline end (origin or extremity) is "set" to -1 at one end and set to 1 or 2 at its other end the
  action is **ambiguous**.

The way to compute the impact of the action has also been adjusted to reflect these changes.

In the table below we try to summarize all the possible actions and their impact on the powerline.
This table is made considering that "`LINE_ID`" is an id of a powerline and "`SUB_OR`" is the id of the origin of the
substation. If a status is 0 it means the powerlines is disconnected, if the status is 1 it means it is connected.

=============================================  ================  ============   ====================   ====================
action                                         original status   final status   substations affected   line status affected
=============================================  ================  ============   ====================   ====================
{"set_line_status": [(LINE_ID, -1)]}           1                 0              None                    LINE_ID
{"set_line_status": [(LINE_ID, +1)]}           1                 1              None                    LINE_ID
{"set_line_status": [(LINE_ID, -1)]}           0                 0              None                    LINE_ID
{"set_line_status": [(LINE_ID, +1)]}           0                 1              None                    LINE_ID
{"change_line_status": [LINE_ID]}              1                 0              None                    LINE_ID
{"change_line_status": [LINE_ID]}              0                 1              None                    LINE_ID
{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}}  1                 0              None                    INE_ID
{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}}  0                 0              SUB_OR                  None
{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}}   1                 1              SUB_OR                  None
{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}}   0                 1              None                    LINE_ID
{"change_bus": {"lines_or_id": [LINE_ID]}}     1                 1              SUB_OR                  None
{"change_bus": {"lines_or_id": [LINE_ID]}}     0                 0              SUB_OR                  None
=============================================  ================  ============   ====================   ====================

This has other impacts. In grid2op there is a convention that if an object is disconnected,
then it is assigned to bus "-1". For a powerline this entails that a status changed affects the bus of

As we explained in the previous paragraph, some action on one end of a powerline can reconnect a
powerline or disconnect it. This means they modify the bus of **both** the extremity of the powerline.

Here is a table summarizing how the buses are impacted. We denoted by "`PREVIOUS_OR`" the last bus at which
the origin end of the powerline was connected and "`PREVIOUS_EX`" the last bus at which the extremity end of the
powerline was connected. Note that for clarity when something is not modified by the action we decided to write on
the table "not modified" (this entails that after this action, if the powerline is connected then "new origin bus" is
"`PREVIOUS_OR`" and "new extremity bus" is "`PREVIOUS_EX`"). We remind the reader that "-1" encode for a
disconnected object.

=============================================  ================  ============   ==============  ========================
action                                         original status   final status   new origin bus  new extremity bus
=============================================  ================  ============   ==============  ========================
{"set_line_status": [(LINE_ID, -1)]}           1                 0              -1              -1
{"set_line_status": [(LINE_ID, +1)]}           1                 1              Not modified    Not modified
{"set_line_status": [(LINE_ID, -1)]}           0                 0              Not modified    Not modified
{"set_line_status": [(LINE_ID, +1)]}           0                 1              PREVIOUS_OR     PREVIOUS_EX
{"change_line_status": [LINE_ID]}              1                 0              -1              -1
{"change_line_status": [LINE_ID]}              0                 1              PREVIOUS_OR     PREVIOUS_EX
{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}}  1                 0              -1              -1
{"set_bus": {"lines_or_id": [(LINE_ID, -1)]}}  0                 0              Not modified    Not modified
{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}}   1                 1              2               Not modified
{"set_bus": {"lines_or_id": [(LINE_ID, 2)]}}   0                 1              2               PREVIOUS_EX
{"change_bus": {"lines_or_id": [LINE_ID]}}     1                 1              \*              Not modified
{"change_bus": {"lines_or_id": [LINE_ID]}}     0                 0              Not modified    Not modified
=============================================  ================  ============   ==============  ========================

\* means that this bus is affected: if it was on bus 1 it moves on bus 2 and vice versa.

.. _action-module-converter:

Easier actions manipulation
----------------------------
The action class presented here can be quite complex to apprehend, especially for a machine learning algorithm.

It is possible to use the :class:`grid2op.Converter` class for such purpose. You can have a look at the dedicated
documentation.


Detailed Documentation by class
-------------------------------

.. automodule:: grid2op.Action
    :members:
    :private-members:
    :special-members:
    :autosummary:


.. include:: final.rst

