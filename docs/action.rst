.. currentmodule:: grid2op.Action

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
that the :class:`grid2op.Backend` class will be able to understand it. If you don't, your extension will not affect the
underlying powergrid. Indeed a :class:`grid2op.Backend` will call the :func:`BaseAction.__call__` method and should
understands its return type.


The :class:`BaseAction` and all its derivatives also offer some usefull inspection utilities:

  - :func:`BaseAction.__str__` prints the action in a format that gives usefull information on how it will affect the powergrid
  - :func:`BaseAction.effect_on` returns a dictionnary that gives information about its effect.

From :class:`BaseAction` inherit in particular the :class:`PlayableAction`, the base class of all action that
players are allowed to play.

Finally, :class:`BaseAction` class define some strict behavior to follow if reimplementing them. The correctness of each
instances of BaseAction is assessed both when calling :func:`BaseAction.update` or with a call to
:func:`BaseAction._check_for_ambiguity` performed for example by the Backend when it must implement its effect on the
powergrid through a call to :func:`BaseAction.__call__`

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

