.. currentmodule:: grid2op.Rules

Rules of the Game
===================================

Objectives
-----------
The Rules module define what is "Legal" and what is not. For example, it can be usefull, at the beginning of the
training of an :class:`grid2op.Agent.BaseAgent` to loosen the rules in order to ease the learning process, and have
the agent focusing more on the physics. When the agent is performing well enough, it is then possible to make the
rules more and more complex up to the target complexity.

Rules includes:

- checking the number of powerline that can be connected / disconnected at a given time step
- checking the number of substations for which the topology can be reconfigured at a given timestep

If an action "break the rules" it is replaced by a do nothing. Note that the Rules of the game is different
from the concept of Ambiguous Action.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Rules
    :members:
    :private-members:
    :special-members:
    :autosummary:

.. include:: final.rst
