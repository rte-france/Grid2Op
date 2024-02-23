.. _mdp-doc-module:

Dive into grid2op sequential decision process
===============================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------

TODO

Modeling sequential decisions
-------------------------------

TODO


Inputs
~~~~~~~~~~

A simulator
++++++++++++

TODO

B Time Series
++++++++++++++

TODO

Markov Decision process
~~~~~~~~~~~~~~~~~~~~~~~~

Extensions
-----------

Partial Observatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the case in most grid2op environment: only some part of the environment
state at time `t` :math:`s_t` are
given to the agent in the observation at time `t` :math:`o_t`.

More specifically, in most grid2op environment (by default at least), none of the 
physical parameters of the solvers are provided. Also, to represent better
the daily operation in power systems, only the `t`th row :math:`x_t` of the matrix
X is given in the observation :math:`o_t`. The components :math:`X_{t', i}` 
(for :math:`t' > t`) are not given.

Adversarial attacks
~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: explain the model of the environment

Forecast and simulation on future states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO : explain the model the forecast and the fact that the "observation" also
includes a model of the world that can be different from the grid of the environment

Simulator dynamics can be more complex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hide elements from the grid2op environment
++++++++++++++++++++++++++++++++++++++++++

TODO only a part of the grid would be "exposed" in the
grid2op environment.


Contain elements not modeled by grid2op
++++++++++++++++++++++++++++++++++++++++++

TODO: speak about HVDC or "pq" generators, or 3 winding transformers

Contain embeded controls
++++++++++++++++++++++++++++++++++++++++++

TODO for example automatic setpoint for HVDC or limit on Q for generators

Time domain simulation
+++++++++++++++++++++++

TODO: we can plug in simulator that solves more
accurate description of the grid and only "subsample"
(*eg* at a frequency of every 5 mins) provide grid2op
with some information.

.. include:: final.rst
