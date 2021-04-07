.. currentmodule:: grid2op.Agent
.. _agent-module:

Agent
============

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3


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

To get started coding your agent we encourage you to read the description of the :ref:`action-module` to know how
to implement your action. Don't hesitate to have a look at the :ref:`action-module-converter` for
an easier / higher level action manipulation.

Once you know how to manipulate a powergrid in case of the grid2op framework, you can easily implement an agent
following this example

.. code-block:: python

    import grid2op
    from grid2op.Agent import BaseAgent

    class MyCustomAgent(BaseAgent):
        def __init__(self, action_space, something_else, and_another_something):
            # define here the constructor of your agent
            # here we say our agent needs "something_else" and "and_another_something"
            # to be built just to demonstrate it does not cause any problem to extend the
            # construction of the base class BaseAgent that only takes "action_space" as a constructor
            BaseAgent.__init__(self, action_space)
            self.something_else = something_else
            self.and_another_something = and_another_something

        def act(obs, reward, done=False):
            # this is the only method you need to implement
            # it takes an observation obs (and a reward and a flag)
            # and should return a valid action
            dictionary_describing_the_action = {}  # this can be anything you want that grid2op understands
            my_action = env.action_space(dictionary_describing_the_action)
            return my_action


Detailed Documentation by class
-------------------------------
.. automodule:: grid2op.Agent
    :members:
    :private-members:
    :autosummary:

.. include:: final.rst