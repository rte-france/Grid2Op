.. module:: grid2op
.. _grid2op-module:

Grid2Op module
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

The grid2op module allows to model sequential decision making on a powergrid.

It is modular in the sense that it allows to use different powerflow solver (denoted as "Backend").
It proposes an internal representation
of the data that can be feed to powergrids and multiple classes to specify how it's done.

For example, it is possible to use an "action" to set the production value of some powerplant. But we
also know that it's not possible to do this for every powerplant (for example, asking a windfarm to produce more
energy is not possible: the only way would be to increase the speed of the wind). It is possible to implement
these kind of restrictions in this "game like" environment.

Today, the main usage of this platform is to serve as a computation engine for the `L2RPN <www.l2rpn.chalearn.com>`_
competitions.

This platform is still under development. If you notice a bug, let us know with a github issue at
`Grid2Op <https://github.com/rte-france/Grid2Op>`_

.. note:: Grid2op do not model any object on the powergrid. It has no internal modeling of the equations of the
    grids, or what kind of solver you need to adopt.

    On the other hand, grid2op aims at representing the grid in a relatively "high level" point of view: it knows
    which "elements" are connected to which, which is the production of this or that generators or how much power
    is consumed by this load.

    But under no circumstances, grid2op will expose some specific modeling of a powergrid. Such modeling are
    only made in the Backend.

    A somewhat relatively "accurate" view of what grid2op is to answer questions such as "if I had put a sensor at this
    location - *eg* right next to a powerplant - what would this sensor would have given ? It then takes care
    of exporting these data to a "format" to the entities acting on the grid.


Objectives
-----------
The primary goal of grid2op is to model decision making process in power systems. Indeed, we believe that developing
new flexibilities on the grid would make the
"energy transition" an easier, less costly process.

It allows fast development of new "methods" that are able to "take decisions" on a powergrid and assess how
well these controllers perform (see section `Controlling the grid`_ for more information about the "controls") .

Thanks to a careful separation between:

- the data used to represent how the powergrid is evolving (represented by the `Chronics`)
- the solver that is able to compute the state of the grid (represented by the `Backend`)
- the controller / agent that takes action on the powergrid (represented by the `Agent`)

All bound together thanks to the :class:`grid2op.Environment` module.

Grid2op attempts also to make the development of new control methods as easy as possible: it is relatively simple
to generate data and train agent on them and to use a fast (but less precise powerflow) while trying
to develop new state of the art methods. While still being usable in a "closer to reality" setting where data
can come from real grid state that happened in the past and the solver is as accurate as possible. You can switch
from one to the other almost effortlessly.

For a more detailed description, one can consult the
`Reinforcement Learning for Electricity Network Operation <https://arxiv.org/abs/2003.07339>`_
paper.

Though grid2op has been primarily developed for the L2RPN competitions series, it is more general. Its modularity
can also help developing and benchmarking new powerflow solvers for example.

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
- the storage units (batteries or pumped storage) that allows to produce some energy / absorb some energy from the
  powergrid when needed.

Other "flexibilities" (ways to act on the grid) are coming soon (-:

.. note:: We wanted to emphasize the particularity of the problem proposed in grid2Op.
    Indeed, the objective is to act on a graph (observation space = a graph, action space = modification of this graph).

    As opposed to many graph related problems addressed in the literature, we do not try to find some properties of a
    dataset represented as one (or many) graph(s).

    Controlling a powergrid rather means to find a graph that meets some properties (**eg** all weights on all
    edges **aka** the flows on the powerlines, must be bellow some threshold **aka** the thermal limits - **NB** a
    solver uses some physical laws to compute these "weights" from the amount of power produced / absorbed in
    different part of the grid where generators and loads are connected).

What is modeled in an grid2op environment
-----------------------------------------
The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of
temporal injections (productions and consumptions) or maintenance / hazards for discretized
time-steps (usually there is the equivalent of *5* minutes between two consective steps).

Say a powergrid is represented as a graph with:

- the edges being the powerlines (and transformers)
- the nodes being the "bus" (a bus is the power system terminology to denotes the "things" (aka nodes) that are
  connected by the edges)

.. note:: Grid2op does not explicitly model the "graph" of the grid as a "graph" structure. For performances, it is
    represented as a vector, as explained in paragraph ":ref:`topology-pb-explained`". To be exhaustive, the way to
    map this graph to this vector is explained in the page ":ref:`create-backend-module`" (though this page is
    really detailed and has too much information for most grid2op usage).

    Some functions have been coded to retrieve the state, as a "graph" (more precisely a square matrix). These methods
    are described in the section ":ref:`observation_module_graph`" of the Observation module.

This graph has some properties:

- some buses are labeled "generators" that produces a certain amount of power
- some buses are labeled "loads" that consumes a certain amount of power  (**NB** a bus can be both a generator
  and a load, in this case both the production and the demand should be met at his node)
- all edges have some  "weights": some physical laws (*eg* conservation of energy or more specifically
  `Kirchoff Circuits Laws`), that cannot be altered (and are computed by the `Backend`), induced some flows on
  the powerline that can be represented as "weights" on this graph
- it is dynamic: at different steps, the graph can be different, for example, it is possible to have a "node" with
  load 1, load 2, line 1 and line 2 and a given step, and to "split" this node in two to have, at another step
  load 1 and line 2 on a "node" and "load 2" and "line 1" on a different node (and the other way around).

This graph has some constraints:

- the total generation (sum of production of all generator) should be exactly equal to the
  total demand (sum of consumption of all loads) and the power losses (due to the heating of the powerlines for
  example)
- the generators should always be connected to the grid, otherwise this is a blackout
- the loads should always be connected to the grid, otherwise this is a blackout
- the graph of the grid should be `connected` (made of one unique connex component): otherwise the condition number
  1 above (sum production = sum load + sum losses) will not be met in each of the independant subgraph, most likely.
- there exist a solution to the `Kirchoff Circuits Laws`

For more information on this "graph" and the way to retrieve it
in different format, you can consult the page :ref:`gridgraph-module` of the documentation.

The whole grid2op ecosystem aims at modeling the evolution of a "controller" that is able to make sure the
"graph of grid", at all time meets all the constraints.

More concretely a grid2op environment models "out of the box":

- the mechanism to "implement" a control on the grid (with a dedicated `action` module) that can be used by any
  `Agent`, which takes some decisions to maintain the grid in security
- time series of loads and productions: which represents the evolution of the power injected / withdrawn
  at each bus of the grid, at any time (**NB** the `Agent` do not see the future, it means that it cannot have an
  exact value for each of the loads in the future, but can only observe the current sate)
- a mechanism (that can be implemented using different solver) to compute the flows based on the injections (which
  among of power is produced at each nodes) and the topology (graph of the grid)
- the automatic disconnection of powerlines if there are on overflow for too long (known as "time overcurrent (TOC)" see
  this article for more information
  `overcurrent <https://en.wikipedia.org/wiki/Power_system_protection#Overload_and_back-up_for_distance_(overcurrent)>`_ )
  Conceptually this means the environment remember for how long a powergrid is in "overflow" and disconnects it
  if needed. **NB** This is an **emulation** of what happen on the grid, in case you use a Backend that do not have
  this feature (for example if you use static / steady state powerflow). This emulation might not be necessary (and
  less "realistic" if you use a time domain simulator)
- the disconnection of powerlines if the overflow is too high (known as "instantaneous overcurrent" see the same
  wikipedia article). This means from one step to another, a given powerline can be disconnected if too much
  flow goes through it. **NB** This is an **emulation** of what happen on the grid, in case you use a Backend that do not have
  this feature (for example if you use static / steady state powerflow). This emulation might not be necessary (and
  less "realistic" if you use a time domain simulator)
- the maintenance operations: if there is a planned maintenance, the environment is able to disconnect a powerline
  for a given amount of steps and preventing its reconnection. There are information about such planned event
  that are given to the controller.
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
  constraints) the production of each generators. Knowing these information at any time steps, the powergrid state
  must satisfy the `Kirchhoff's circuit laws <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_ .
- stochastic environments: in all environment, you don't know the future, which makes it a "Partially
  Observable" environments (if you were in a maze, you would **not** see "from above" but rather see "at the first
  person": only seeing in front of you).
  Environments can be "even more stochastic" if there are hazards or even adversarial: a malicious agent can take
  attacks targeted to endanger your policy.
- with both **continuous and discrete observation space**: some part of the observation are continuous (for example
  the amount of flow on a given powerline, or the production of this generator) and some are discrete (
  for example the status - connected / disconnected - of a powerline, or how long this powerline
  has been in overflow etc.)
- with **both continuous and discrete action space**: the preferred type of action is the topology, which is
  represented as a discrete type of action (*eg* you can either connect / disconnect a powerline) but there exist
  also some continuous actions (for example you can adjust in real time the production of a set of generators)
- dynamic graph manipulation: power network can be modeled as graphs. In these environments both the observation
  **and the action** are focused on graph. The observation contains the complete state of the grid, including
  the "topology" (you can think of it a its graph) and actions are focused on adapting this graph to make
  the grid as robust and secure as possible. **NB** As opposed to most problem in the literature, where
  you need to find some properties (label of of the edges or the nodes, etc.) in grid2op you need
  to find a graph that meets some properties: find a graph that meets constraints on its edges and its nodes.
- strong emphasis on **safety** and **security**: power system are highly critical system (who would want to
  short circuit a powerplant? Or causing a blackout preventing an hospital to cure the patients?) and as such it is
  critical that the controls keep the powergrid safe in all circumstances.


Modeling the interaction with operators
----------------------------------------

In real grid it is likely that human operators will still be in command for at
least a certain number of tasks (if not all !) including controlling flows in 
the grid.

To that end, it is important that the "AI" / "Artificial Agent" / "Algorithm"
(what is modeled by the `Agent` in grid2Op) collaborates well with human.

In grid2op there are two main concepts related to "human machine interaction":
- alarm: used for example in "l2rpn_icaps_2021" environment.
- alerts: used for example in "l2rpn_idf_2023" environment.

Alarm feature
+++++++++++++++

This section might or might not be updated depending on the time at our disposal...

.. _grid2op-alert-module:

Alert feature
+++++++++++++++

In a "human / machine" collaboration it is important that the machine tells the human when 
it is not fully confident on its ability to handle the grid for `xxx` amount of time (`xxx` is 
in fact `env.parameters.ALERT_TIME_WINDOW`).

.. note::
  At time of writing only the env "l2rpn_idf_2023" supports this feature.

In our formulation, we ask the agent to send alert (through the action see :attr:`grid2op.Action.BaseAction.raise_alert`)
at each step. 

An alert will concern a single powerline (say powerline `i`) And each of this alert will mean:

- if no alert (for line `i`) is raised at this step: "Me, the agent, can handle the grid for `env.parameters.ALERT_TIME_WINDOW` 
  steps even if powerline `i` is disconnected" (
  the agent tell the human there will not be any game over for at least `env.parameters.ALERT_TIME_WINDOW` steps even if
  the opponent attacks line `i`)
- if an alert (for line `i`) is raised it means the opposite. The agent "think" that, if powerline `i` is attacked, then
  it will probably game over before `env.parameters.ALERT_TIME_WINDOW` steps pass.

.. note:: 
  This entails that the alerts can only be evaluated when an opponent attacks a powerline. If a powerline is attacked, then 
  the environment "waits" for `env.parameters.ALERT_TIME_WINDOW` and there is 4 cases:

  - agent survived and sent an alert on line `i` (just before this line was attacked): 
    this is not a behaviour that we want to incite, reward is -1.
  - agent survived and did not sent an alert on line `i` (just before this line was attacked): this is a correct
    behaviour, the reward is +1.
  - agent "games over" during the `env.parameters.ALERT_TIME_WINDOW` and sent an alert 
    on line `i` (just before this line was attacked): though the game over should be avoided at all cost, this 
    is the "expected" behaviour for the alert and in this case the reward is +2. 
  - agent "games over" during the `env.parameters.ALERT_TIME_WINDOW` and did not sent an alert 
    on line `i` (just before this line was attacked): the game over should be avoided at all cost and this
    is NOT the correct behaviour (agent should have told it was not able to conduct the grid) for the alert
    agent is heavily penalized with a score of -10.

  (Above this entails that the :class:`grid2op.Reward.AlertReward` is used)

.. danger::
  The "reward" in the warning above is only relevant for the "alert" part. Note that for L2RPN competition, the 
  real goal is still to oeprate the grid for as long as possible. There is an heavy penalty in these competitions
  if an agent "games over" before the end.
   

To model all that you have, at your disposal, in the observation, the attributes:

- :attr:`grid2op.Observation.BaseObservation.active_alert`
- :attr:`grid2op.Observation.BaseObservation.time_since_last_alert`
- :attr:`grid2op.Observation.BaseObservation.alert_duration`
- :attr:`grid2op.Observation.BaseObservation.total_number_of_alert`
- :attr:`grid2op.Observation.BaseObservation.time_since_last_attack`
- :attr:`grid2op.Observation.BaseObservation.was_alert_used_after_attack`
- :attr:`grid2op.Observation.BaseObservation.attack_under_alert`

And at each step, an agent can raise an alert with the action property :attr:`grid2op.Action.BaseAction.raise_alert`.

We also added a dedicated reward for this feature: :class:`grid2op.Reward.AlertReward`

.. note::
  Once raised, an alert on a powerline is valid for the next step only. You need to re raise it at the next
  step if you want it to lasts in time more than one steps.

.. note::
  For an alert to be taken into account it should be raised BEFORE a powerline is attacked. It means that
  if the opponent alread attacked a powerline, it is not useful to raise an alert on said powerline as it 
  will have no impact at all.

Let's take some examples with the following environment and seed:

.. code-block:: python

  import grid2op
  import numpy as np

  env = grid2op.make("l2rpn_idf_2023", test=True)
  env.seed(0)
  obs = env.reset()
  # I know (because programmer know, that's why ;-) )
  # that an opponent will attack powerline 106 at step 14
  # powerline 106 is attackable line id 0 (np.where(type(env).alertable_line_ids) == 106))
  # test the attack in the original config
  for i in range(14):
      obs, reward, done, info = env.step(env.action_space())
      print(obs.current_step, info["opponent_attack_line"], np.where(info["opponent_attack_line"])[0] if info["opponent_attack_line"] is not None else None)


Let's play the same scenario again: same attack, same everything:

.. code-block:: python

  env.seed(0)
  obs = env.reset()
  for i in range(12):
      obs, reward, done, info = env.step(env.action_space())
  # still no attack at this point

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # no attack at this step, the previous action has no impact at all !

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # an attack at this step on line attackable 0

The agent sent an alarm at the right time before the attack. It means that the agent expects to fail, 
after this attack and within the next 12 steps. If it fails between "now" and "12 steps from now" 
reward associated with alert will be positive else it will be negative.


Let's replay again the same scenario again: same attack, same everything:

.. code-block:: python

  env.seed(0)
  obs = env.reset()
  for i in range(12):
      obs, reward, done, info = env.step(env.action_space())
  # still no attack at this point

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # no attack at this step, the previous action has no impact

  act = env.action_space()
  obs, reward, done, info = env.step(act)  # an attack at this step on line attackable 0

The agent did not sent any alarm right before the attack on "attackable line 0".
It means that the agent expects to survive after this attack for at least the next 12 steps. 
If it fails between "now" and "12 steps from now" reward associated with alert will highly
negative (this is the situation where the agent should have told the human operator "help me").


Let's replay again (again ?) the same scenario again: same attack, same everything:

.. code-block:: python

  env.seed(0)
  obs = env.reset()
  for i in range(12):
      obs, reward, done, info = env.step(env.action_space())
  # still no attack at this point

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # no attack at this step, the previous action has no impact

  act = env.action_space({"raise_alert": [1]})
  obs, reward, done, info = env.step(act)  # an attack at this step on line attackable 0
  print(obs.attack_under_alert[0])

The agent raised an alert at the right time, but not on the attacked line. This means that
the agent is confident that it can handle the attack (on attackable line 0) but that
it "thinks" it would be in trouble if attackble line 1 had been attacked instead. This is
the same case as just above: the "thinks" it will survive for at least the next 12 steps.

And now let's replay one last time the same everything:

.. code-block:: python

  env.seed(0)
  obs = env.reset()
  for i in range(12):
      obs, reward, done, info = env.step(env.action_space())
  # still no attack at this point

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # no attack at this step, the previous action has no impact

  act = env.action_space()
  obs, reward, done, info = env.step(act)  # an attack at this step on line attackable 0

  act = env.action_space({"raise_alert": [0]})
  obs, reward, done, info = env.step(act)  # the attack continues at this step on line attackable 0

In this case, the agent sents the alert AFTER the incident (disconnection of powerline 0)
occurs. The sent alert will not be used by grid2op. It will be equivalent as the 2 cases above:
the agent is confident in its ability to handle the grid for the next 12 steps even if it sends an
alert (when the attack is happening)


Disclaimer
-----------

Grid2op is a research testbed platform, it has not been tested in "production" context

Going further
--------------
To get started into the grid2op ecosystem, we made a set of notebooks
that are available, without any installation thanks to
`Binder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_ . Feel free to visit the "getting_started" page for
more information and a detailed tour about the issue that grid2op tries to address.

.. note:: As of writing (december 2020) most of these notebooks focus on the "agent" part of grid2op. We would welcome
    any contribution to better explain the other aspect of this platform.

.. include:: final.rst
