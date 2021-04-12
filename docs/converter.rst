.. currentmodule:: grid2op.Converter

Converters
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
In this module of grid2op, the "converters" are defined.

A converter is a specific class of :class:`grid2op.Action.ActionSpace` (ie of BaseAction Space) that allows the
agent to manipulate this action to have a different representation of it.

For example, suppose we are dealing with :class:`grid2op.Action.TopologyAndDispatchAction` (only manipulating the
graph of the powergrid). This is a discrete "action space". Often, it's custom to deal with such action space by
enumerating all actions, and then assign to all valid actions a unique ID.

This can be done easily with the :class:`IdToAct` class.

More concretely, the diagram of an agent is:

i) receive an observation (in a form of an object of class :class:`grid2op.Observation.BaseObservation`)
ii) implement the :func:`grid2op.Agent.BaseAgent.act` taking as input an
    :class:`grid2op.Observation.BaseObservation` and
    returning an :class:`grid2op.Action.BaseAction`
iii) this :class:`grid2op.Action.BaseAction` is then digested by the environment

Introducing some converters lead to the following:

i) receive an observation (:class:`grid2op.Observation.BaseObservation`)
ii) the transformer automatically (using :func:`Converter.convert_obs`) to a `transformed observation`
iii) implement the function :func:`grid2op.Agent.AgentWithConverter.my_act` that takes as input
     a `transformed observation` and returns an `encoded action`
iv) the transformer automatically transforms back the `encoded action` into a proper
    :class:`grid2op.Action.BaseAction`
v) this :class:`grid2op.Action.BaseAction` is then digested by the environment

This simple mechanism allows people to focus on iii) above (typically implemented with artificial neural networks)
without having to worry each time about the complex representations of actions and observations.

More details and a concrete example is given in the documentation of the class
:class:`grid2op.Agent.AgentWithConverter`.

Some examples of converters are given in :class:`IdToAct` and :class:`ToVect`.


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Converter
    :members:
    :private-members:
    :autosummary:

.. include:: final.rst
