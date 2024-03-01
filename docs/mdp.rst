.. include:: special.rst

.. _mdp-doc-module:

Dive into grid2op sequential decision process
===============================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------

The goal of this page of the documentation is to provide you with a relatively extensive description of the 
mathematical model behind grid2op.

Grid2op is a software whose aim is to make experiments on powergrid, mainly sequential decision making,
as easy as possible. 

We chose to model this sequential decision making probleme as a 
"*Markov Decision Process*" (MDP) and one some cases
"*Partially Observable Markov Decision Process*" (POMDP) or 
"*Constrainted Markov Decision Process*" (CMDP) and (work in progress) even
"*Decentralized (Partially Observable) Markov Decision Process*" (Dec-(PO)MDP).

General notations
~~~~~~~~~~~~~~~~~~~~

There are different ways to define an MDP. In this paragraph we introduce the notations that we will use.

In an MDP an "agent" / "automaton" / "algorithm" / "policy" takes some action :math:`a_t \in \mathcal{A}`. This
action is processed by the environment and update its internal state from :math:`s_t \in \mathcal{S}` 
to :math:`s_{t+1} \in \mathcal{S}` and 
computes a so-called *reward* :math:`r_{t+1} \in [0, 1]`. 

.. note::
    By stating the dynamic of the environment this way, we ensure the "*Markovian*" property: the
    state :math:`s_{t+1}` is determined by the knowledge of the previous state :math:`s_{t}` and the
    action :math:`a_{t}`

This tuple
:math:`(s_t, r_t)` is then given to the "agent" / "automaton" / "algorithm" which in turns produce the action :math:`a_{t+1}`

.. note::
    More formally even, everything written can be stochastic:

    - :math:`a_t \sim \pi_{\theta}(s_t)` where :math:`\pi_{\theta}(\cdot)` is the "policy" parametrized by
      some parameters :math:`\theta` that outputs here a probability distribution (depending on the 
      state of the environment :math:`s_t`) over all the actions `\mathcal{A}`
    - :math:`s_{t+1} \sim \mathcal{L}_S(s_t, a_t)` where :math:`\mathcal{L}_S(s_t, a_t)` is a probability distribution
      over :math:`\mathcal{S}` representing the likelyhood if the "next state" given the current state and the action
      of the "policy"
    - :math:`r_{t+1} \sim \mathcal{L}_R(s_t, s_{t+1}, a_t)` is the reward function indicating "how good"
      was the transition from :math:`s_{t}` to :math:`s_{t+1}` by taking action :math:`a_t`


This alternation :math:`\dots \to a \to (s, r) \to a \to \dots` is done for a certain number of "steps" called :math:`T`.

We will call the list :math:`s_{1} \to a_1 \to (s_2, r_2) \to \dots \to a_{T-1} \to (s_{T}, r_T)`
an "**episode**".

Formally the knowledge of:

- :math:`\mathcal{S}`, the "state space"
- :math:`\mathcal{A}`, the "action space"
- :math:`\mathcal{L}_s(s, a)`, sometimes called "transition kernel", is the probability 
  distribution (over :math:`\mathcal{S}`) that gives the next
  state after taking action :math:`a` in state :math:`s`
- :math:`\mathcal{L}_r(s, s', a)`, sometimes called "reward kernel",
  is the probability distribution (over :math:`[0, 1]`) that gives
  the reward :math:`r` after taking action :math:`a` in state :math:`s` which lead to state :math:`s'`
- :math:`T \in \mathbb{N}^*` the maximum number of steps for an episode

Defines a MDP. We will detail all of them in the section :ref:`mdp-def` bellow.

In grid2op, there is a special case where a grid state cannot be computed (either due to some physical infeasibilities
or because the resulting state would be irrealistic). This can be modeled relatively easily in the MDP formulation 
above if we add a "terminal state" :math:`s_{\emptyset}` in the state space :math:`\mathcal{S}_{new} := \mathcal{S} \cup \left\{ s_{\emptyset} \right\}`: and add the transitions: 
:math:`\mathcal{L}_s(s_{\emptyset}, a) = \text{Dirac}(s_{\emptyset}) \forall a \in \mathcal{A}`
stating that once the agent lands in this "terminal state" then the game is over, it stays there until the 
end of the scenario. 

We can also define the reward kernel in this state, for example with 
:math:`\mathcal{L}_r(s_{\emptyset}, s', a) = \text{Dirac}(0) \forall s' \in \mathcal{S}, a \in \mathcal{A}` and
:math:`\mathcal{L}_r(s, s_{\emptyset}, a) = \text{Dirac}(0) \forall s \in \mathcal{S}, a \in \mathcal{A}` which
states that there is nothing to be gained in being in this terminal set.

Unless specified otherwise, we will not enter these details in the following explanation and take it as
"pre requisite" as it can be defined in general. We will focus on the definition of :math:`\mathcal{S}`, 
:math:`\mathcal{A}`, :math:`\mathcal{L}_s(s, a)` and :math:`\mathcal{L}_r(s, s', a)` by leaving out the
"terminal state".

.. note::
  In grid2op implementation, this "terminal state" is not directly implemented. Instead, the first Observation leading 
  to this state is marked as "done" (flag `obs.done` is set to `True`). 
  
  No other "observation" will be given by 
  grid2op after an observation with `obs.done` set to `True` and the environment needs to be "reset".

  This is consistent with the gymnasium implementation.

The main goal of a finite horizon MDP is then to find a policy :math:`\pi \in \Pi` that given states :math:`s` and reward :math:`r`
output an action :math:`a` such that (*NB* here :math:`\Pi` denotes the set of all considered policies for this
MDP):

.. math::
  :nowrap:

  \begin{align*}
      \min_{\pi \in \Pi}  ~& \sum_{t=1}^T \mathbb{E} r_t \\
      \text{s.t.} ~ \\
                     & \forall t, a_t \sim  \pi (s_{t}) & \text{policy produces the action} \\
                     & \forall t, s_{t+1} \sim \mathcal{L}_S(s_t, a_t) & \text{environment produces next state} \\
                     & \forall t, r_{t+1} \sim \mathcal{L}_r(s_t, a_t, s_{t+1}) & \text{environment produces next reward} \\
  \end{align*}

Specific notations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To define "the" MDP modeled by grid2op, we also need to define some other concepts that will be used to define the 
state space :math:`\mathcal{S}` or transition kernel :math:`\mathcal{L}_s(s, a)` for example.

A Simulator
++++++++++++

We need a so called "simulator". 

Informatically, this is represented by the `Backend` inside the grid2op environment (more information 
about the `Backend` is detailed in the :ref:`backend-module` section of the documentation).

This simulator is able to compute some informations that are part of the state
space :math:`\mathcal{S}` (*eg* flows on powerlines, active production value of generators etc.)
and thus are used in the computation of the transition kernel.

We can model this simulator with a function :math:`\text{Sim}` that takes as input some data from an 
"input space" :math:`\mathcal{S}_{\text{im}}^{(\text{in})}` and result 
in data in :math:`\mathcal{S}_{\text{im}}^{(\text{out})}`.

.. note::
  In grid2op we don't force the "shape" of :math:`\mathcal{S}_{\text{im}}^{(\text{in})}`, including
  the format used to read the grid file from the hard drive, the solved equations, the way
  these equations are used. Everything here is "free" and grid2op only needs that the simulator
  (wrapped in a `Backend`) understands the "format" sent by grid2op (through a 
  :class:`grid2op.Action._backendAction._BackendAction`) and is able to expose
  to grid2op some of its internal variables (accessed with the `***_infos()` methods of the backend)


To make a parallel with similar concepts "simulator",
represents the physics as in all `"mujoco" environments <https://gymnasium.farama.org/environments/mujoco/>`_ 
*eg* `Ant <https://gymnasium.farama.org/environments/mujoco/ant>`_ or 
`Inverted Pendulum <https://gymnasium.farama.org/environments/mujoco/inverted_pendulum>`_ . This is the same concept
here excepts that it solves powerflows.

Some Time Series
+++++++++++++++++

Another type of data that we need to define "the" grid2op MDP is the "time series", implemented in the `chronics`
grid2op module documented on the page 
:ref:`time-series-module` with some complements given in the :ref:`doc_timeseries` page as well. 

These time series define what exactly would happen if the grid was a 
"copper plate" without any constraints. Said differently it provides what would each consumer
consume and what would each producer produce if they could all be connected together with 
infinite "bandwith", without any constraints on the powerline etc.

In particular, grid2op supposes that these "time series" are balanced, in the sense that the producers
produce just the right amount (electrical power cannot really be stocked) for the consumer to consume 
and that for each steps. It also supposes that all the "constraints" of the producers.

These time series are typically generated outside of grid2op, for example using `chronix2grid <https://github.com/BDonnot/ChroniX2Grid>`_ 
python package (or anything else).


Formally, we will define these time series as input :math:`\mathcal{X}_t` all these time series at time :math:`t`. These
exogenous data consist of :

- generator active production (in MW), for each generator
- load active power consumption (in MW), for each loads
- load reactive consumption (in MVAr), for each loads
- \* generator voltage setpoint / target (in kV)

.. note::
  \* for this last part, this can be adapted "on demand" by the environment through the `voltage controler` module.
  But for the sake of modeling, this can be modeled as being external / exogenous data.

And, to make a parrallel with similar concept in other RL environment, these "time series" can represent the layout of the maze
in pacman, the positions of the platforms in "mario-like" 2d games, the different turns and the width of the route in a car game etc. 
This is the "base" of the levels in most games.

Finally, for most released environment, a lof of different :math:`\mathcal{X}` are available. By default, each time the 
environment is "reset" (the user want to move to the next scenario), a new :math:`\mathcal{X}` is used (this behaviour 
can be changed, more information on the section :ref:`environment-module-chronics-info` of the documentation).

.. _mdp-def:

Modeling sequential decisions
-------------------------------

As we said in introduction of this page, we will model a given scenario in grid2op. We have at our disposal:

- a simulator, which is represented as a function :math:`\text{Sim} : \mathcal{S}_{\text{im}}^{(\text{in})} \to \mathcal{S}_{\text{im}}^{(\text{out})}`
- some time series :math:`\mathcal{X} = \left\{ \mathcal{X}_t \right\}_{1 \leq t \leq T}`

And we need to define the MDP through the definition of :

- :math:`\mathcal{S}`, the "state space"
- :math:`\mathcal{A}`, the "action space"
- :math:`\mathcal{L}_s(s, a)`, sometimes called "transition kernel", is the probability 
  distribution (over :math:`\mathcal{S}`) that gives the next
  state after taking action :math:`a` in state :math:`s`
- :math:`\mathcal{L}_r(s, s', a)`, sometimes called "reward kernel",
  is the probability distribution (over :math:`[0, 1]`) that gives
  the reward :math:`r` after taking action :math:`a` in state :math:`s` which lead to state :math:`s'`

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


Some constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


.. include:: final.rst
