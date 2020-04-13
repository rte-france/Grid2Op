.. currentmodule:: grid2op.Action

Action Module
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


Detailed Documentation by class
-------------------------------

.. automodule:: grid2op.Action
    :members:
    :private-members:
    :special-members:
    :autosummary:


.. include:: final.rst

