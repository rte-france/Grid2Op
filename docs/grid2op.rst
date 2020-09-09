.. module:: grid2op
.. _grid2op-module:

Grid2Op module
===================================
The grid2op module allows to model sequential decision making on a powergrid.

It is modular in the sense that it allows to use different powerflow solver. It proposes an internal representation
of the data that can be feed to powergrids and multiple classes to specify how it's done.

For example, it is possible to use an "action" to set the production value of some powerplant. But we
also know that it's not possible to do this for every powerplant (for example, asking a windfarm to produce more
energy is not possible: the only way would be to increase the speed of the wind). It is possible to implement
these kind of restrictions in this "game like" environment.

Today, the main usage of this platform is to serve as a computation engine for the `L2RPN <www.l2rpn.chalearn.com>`_
competitions.

This platform is still under development. If you notice a bug, let us know with a github issue at
`Grid2Op <https://github.com/rte-france/Grid2Op>`_

Objectives
-----------
The primary goal of grid2op is to model decision making process in power systems. Indeed, we believe that developing
new flexibilities on the grid would make the
"energy transition" an easier, less costly process.

It allows fast development of new "methods" that are able to "take decisions" on a powergrid and assess how
well these controllers perform (see section `Controlling the grid`_ for more information about the "controls") .

Thanks to a careful separation between:

- the data used to represent the powergrid
- the solver that is able to compute the state of the grid
- the controller / agent that takes action on the powergrid.

All bound together thanks to the :class:`grid2op.Environment` module.

Grid2op attempts also to make the development of new control methods as easy as possible: it is relatively simple
to generate fake data and train agent on them and to use a fast (but less precise powerflow) while trying
to develop new state of the art methods. While still being usable in a "closer to reality" setting where data
can come from real grid state that happened in the past and the solver is as accurate as possible. You can switch
from one to the other almost effortlessly.

For a more detailed description, one can consult the
`Reinforcement Learning for Electricity Network Operation <https://arxiv.org/abs/2003.07339>`_
paper.

Controlling the grid
--------------------
Modeling all what happens in the powergrid would be an extremely difficult task. Grid2op focusing on controls
that could be done today by a human (happening with **a frequency of approximately the minute**). It does not
aim at simulation really high frequency control that are often automatic today. That being said, such controls
can be taken into account by grid2op if the :class:`grid2op.Backend` allows it.

The main focus of grid2op is to make easy to use of **the topology** to control the flows of the grid.
In real time, it is possible to reconfigure the "topology" of the grid (you can think about it
by the action on changing the graph of the power network). Such modifications are highly non linear
and can have a really counter intuitive impact and we believe they are under used by industry and are under studied
by academics at the moment
(feel free to visit the notebooks `0_Introduction.ipynb`,
`0_SmallExample.ipynb` or the `IEEE BDA Tutorial Series.ipynb` of the official
`grid2op github repository <https://github.com/BDonnot/Grid2Op/tree/master/getting_started>`_ for more information)

Along with the topology, grid2op allows easily to manipulate (and thus control):

- the voltages: by manipulating shunts, or by changing the setpoint value of the generators
- the active generation: by the use of the "redispatching" action.

Other "flexibilities" are coming soon (-:


What is modeled in an grid2op environment
-----------------------------------------
The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of
temporal injections (productions and consumptions) or maintenance / hazards for discretized
timesteps.

More concretely a grid2op environment models "out of the box":

- the mechanism to "implement" a control on the grid, load the next chronics data and compute the appropriate
  state of the power network
- the disconnection of powerlines if there are on overflow for too long (known as "time overcurrent (TOC)" see
  this article for more information
  `overcurrent <https://en.wikipedia.org/wiki/Power_system_protection#Overload_and_back-up_for_distance_(overcurrent)>`_ )
  Conceptually this means the environment remember for how long a powergrid is in "overflow" and disconnects it
  if needed.
- the disconnection of powerlines if the overflow is too high (known as "instantaneous overcurrent" see the same
  wikipedia article). This means from one step to another, a given powerline can be disconnected if too much
  flow goes through it
- the maintenance operations: if there is a planned maintenance, the environment is able to disconnect a powerline
  for a given amount of steps and preventing its reconnection. There are information about such planned event
  that are given to the control
- hazards / unplanned outages / attacks: another issue on power system is the fact that sometimes, some powerline
  get disconnected in a non planned manner. For example, a tree can fall on a powerline, the grid might suffer
  a cyber attack etc. This can also be modeled by grid2op.
- prevent the action on some powerlines: whether it is to model the fact in reality it is not possible to always
  act on the same equipment or because some powerline are out of service (because of an attack, a maintenance
  or because it needs to be repaired), grid2op can model the impossibility
  of acting on a given powerline
- prevent the action on some substations: for the same reasons, sometimes you cannot act on given part of
  the network, preventing you to do some topological actions.
- voltage control: though it is not the main focus of the current platform, grid2op can model automatons that
  can take voltage corrective measures (in the near future we think of adding some protection monitoring
  voltage violation too).
- non violation of generator physical constraints: in real life, generator cannot produce too little nor too much
  (we speak about `gen_pmin` and `gen_pmax`) nor their production can vary too much between consecutive
  steps (this is called `gen_max_ramp_down` and `gen_max_ramp_up`)
- stops the game if the grid is in a too bad shape. This can happen if a load or a generator has been disconnected,
  or if some part of the grid is "islanded" (the graph representing the power network is not connex) or if there is
  no feasible solution to the power system equations

Here are a summary of the main modules:

=============================  =========================================================================================
Module Name                    Main usage
=============================  =========================================================================================
:class:`grid2op.Environment`   Implements all the mechanisms described above
:class:`grid2op.Chronics`      In charge of feeding the data (loads, generations, planned maintenance, etc.) to the Environment
:class:`grid2op.Backend`       Carries out the computation of the powergrid state
:class:`grid2op.Agent`         The controller, in charge of managing the safety of the grid
:class:`grid2op.Action`        The control send by the Agent to the Environment
:class:`grid2op.Observartion`  The information sent by the Environment to the Agent, represents the powergrid state as seen by the Agent
:class:`grid2op.Opponent`      Is present to model the unplanned disconnections of powerline
:class:`grid2op.Rules`         Computes whether or not an action is "legal" at a given time step
:class:`grid2op.Parameters`    Store the parameters that defines for example, on which case an action is legal, or how long a powerline can stay on overflow etc.
=============================  =========================================================================================

Properties of this environments
-------------------------------
The grid2op environments have multiple shared properties:

- highly constrained environments: these environments obey physical laws. You cannot directly choose how much
  power flow on a given powerline, what you can do it choosing the "graph" of the power network and (under some
  constraints) the production of each generators. Knowning that at any time steps, the powergrid state
  must satisfy the `Kirchhoff's circuit laws <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_ .
- stochastic environments: in all environment, you don't know fully the future, which makes it a "Partially
  Observable" environment (if you were in a maze, you would not see "from above" but rather see "at the first
  person"). Environment can be "even more stochastic" if there are hazards / attack on the powergrid.
- with both **continuous and discrete observation space**: some part of the observation are continuous (for example
  the amount of flow on a given powerline, or the production of this generator) and some are discrete (
  for example the status - connected / disconnected - of a powerline, or how long this powerline
  has been in overflow etc.)
- with **both continuous and discrete action space**: the preferred type of action is the topology, which is
  represented as a discrete type of action (you can either connect / disconnect a powerline) but there exist
  also some continuous action (for example you can adjust in real time the production of a set of generators)
- dynamic graph manipulation: power network can be modeled as graphs. In these environments both the observation
  **and the action** are focused on graph. The observation contains the complete state of the grid, including
  the "topology" (you can think of it a its graph) and actions are focused on adapting this graph to be as
  robust as possible
- strong emphasis on **safety** and **security**: power system are highly critical system (who would want to
  short circuit a powerplant? Or causing a blackout preventing an hospital to cure the patients?) and such it is
  critical that the controls keep the powergrid safe in all circumstances.

Disclaimer
-----------
Grid2op is a research testbed platform, it shall not be use in "production" for any kind of application.


Going further
--------------
To get started into the grid2op ecosystem, we made a set of notebooks
that are available, without any installation thanks to
`Binder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_ . Feel free to visit the "getting_started" page for
more information and a detailed tour about the issue that grid2op tries to address.

