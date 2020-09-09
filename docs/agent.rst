.. currentmodule:: grid2op.Agent
.. _agent-module:

Agent
============

Objectives
-----------
In this RL framework, an Agent is an entity that acts on the Environment (modeled in grid2op as an object
of class :class:`Environment`). In grid2op such entity is modeled by the :class:`BaseAgent` class.
It can alternatively be named "bot" or "controller" in other literature.

This module presents a few possible :class:`BaseAgent` that can serve either as baseline, or as example on how to
implement such agents. **NB** Stronger baselines are defined in an another repository.

To perform their actions, agent receive two main signals from the :class:`grid2op.Environment`:

  - the :class:`grid2op.Reward.BaseReward` that states how good the previous has been
  - the :class:`grid2op.Observation.BaseObservation` that is a (partial) view on the state of the Environment.

Both these signals can be use to determine what is the best action to perform on the grid. This is actually the main
objective of an :class:`BaseAgent`, and this is done in the :func:`BaseAgent.act` method.


Detailed Documentation by class
-------------------------------
.. automodule:: grid2op.Agent
    :members:
    :private-members:
    :autosummary:

.. include:: final.rst