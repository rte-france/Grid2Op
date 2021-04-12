.. currentmodule:: grid2op.Opponent

Opponent Modeling
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
Power systems are a really important tool today, that can be as resilient as possible to avoid possibly dramatic
consequences.

In grid2op, we chose to enforce this property by implementing an "Opponent" (modeled thanks to the :class:`BaseOpponent`
that can take malicious actions to deteriorate the state of the powergrid and make tha Agent (:class:`grid2op.Agent`)
fail. To make the agent "game over" is really easy (for
example it could isolate a load by forcing the disconnection of all the powerline that powers it). This would not be
fair, and that is why the Opponent has some dedicated budget (modeled with the :class:`BaseActionBudget`).

The class :class:`OpponentSpace` has the delicate role to:
- send the necessary information for the Opponent to attack properly.
- make sure the attack performed by the opponent is legal
- compute the cost of such attack
- make sure this cost is not too high for the opponent budget.


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Opponent
   :members:

.. include:: final.rst