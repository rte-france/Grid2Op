.. currentmodule:: grid2op.Observation

Observation Module
===================================

Objectives
-----------

In a "reinforcement learning" framework, an :class:`grid2op.Agent` receive two information before taking
any action on
the :class:`grid2op.Environment.Environment`. One of them is the :class:`grid2op.Reward.BaseReward` that tells it
how well the past action
performed. The second main input received from the environment is the :class:`BaseObservation`. This is gives the BaseAgent
partial, noisy, or complete information about the current state of the environment. This module implement a generic
:class:`BaseObservation`  class and an example of a complete observation in the case of the Learning
To Run a Power Network (`l2RPN <https://l2rpn.chalearn.org/>`_ ) competition.

Compared to other Reinforcement Learning problems the L2PRN competition allows another flexibility. Today, when
operating a powergrid, operators have "forecasts" at their disposal. We wanted to make them available in the
L2PRN competition too. In the  first edition of the L2PRN competition, was offered the
functionality to simulate the effect of an action on a forecasted powergrid.
This forecasted powergrid used:

  - the topology of the powergrid of the last know time step
  - all the injections of given in files.

This functionality was originally attached to the Environment and could only be used to simulate the effect of an action
on this unique time step. We wanted in this recoding to change that:

  - in an RL setting, an :class:`grid2op.Agent.BaseAgent` should not be able to look directly at the
    :class:`grid2op.Environment.Environment`.
    The only information about the Environment the BaseAgent should have is through the
    :class:`grid2op.Observation.BaseObservation` and
    the :class:`grid2op.Reward.BaseReward`. Having this principle implemented will help enforcing this principle.
  - In some wider context, it is relevant to have these forecasts available in multiple way, or modified by the
    :class:`grid2op.Agent.BaseAgent` itself (for example having forecast available for the next 2 or 3 hours, with
    the Agent able
    not only to change the topology of the powergrid with actions, but also the injections if he's able to provide
    more accurate predictions for example.

The :class:`BaseObservation` class implement the two above principles and is more flexible to other kind of forecasts,
or other methods to build a power grid based on the forecasts of injections.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Observation
    :members:
    :special-members:
    :autosummary:

.. include:: final.rst