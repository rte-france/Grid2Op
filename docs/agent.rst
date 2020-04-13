.. currentmodule:: grid2op.Agent

Agent Module
============

Objectives
-----------
In this RL framework, an :class:`BaseAgent` is an entity that acts on the :class:`Environment`. BaseAgent can alternatively
be named "bot" or "controller" in other literature.

This module presents a few possible :class:`BaseAgent` that can serve either as baseline, or as example on how to
implement such agents.

To perform their actions, agent receive two main signals from the :class:`grid2op.Environment`:

  - the :class:`grid2op.BaseReward` that states how good the previous has been
  - the :class:`grid2op.BaseObservation` that is a (partial) view on the state of the Environment.

Both these signals can be use to determine what is the best action to perform on the grid. This is actually the main
objective of an :class:`BaseAgent`, and this is done in the :func:`BaseAgent.act` method.


Detailed Documentation by class
-------------------------------
.. automodule:: grid2op.Agent
    :members:
    :private-members:
    :autosummary:

.. include:: final.rst