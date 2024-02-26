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

Definitions
~~~~~~~~~~~~

In an MDP an "agent" / "automaton" / "algorithm" / "policy" takes some action :math:`a_t \in \mathcal{A}`. This
action is processed by the environment and update its internal state from :math:`s_t \in \mathcal{S}` 
to :math:`s_{t+1} \in \mathcal{S}` and 
computes a so-called *reward* :math:`r_{t+1} \in [0, 1)`. 

.. note::
    By stating the dynamic of the environment this way, we ensure the "*Markovian*" property: the
    state :math:`s_{t+1}` is determined by the knowledge of the previous state :math:`s_{t}` and the
    action :math:`a_{t}`

.. note::
    More formally even, everything written can be stochastic:

    - :math:`a_t \sim \pi_{\theta}(s_t)` where :math:`\pi_{\theta}(\cdot)` is the "policy" parametrized by
      some parameters :math:`\theta` that outputs here a probability distribution (depending on the 
      state of the environment :math:`s_t`) over all the actions `\mathcal{A}`
    - :math:`s_{t+1} \sim \mathcal{L}(s_t, a_t)` where :math:`\mathcal{L}(s_t, a_t)` is a probability distribution
      over :math:`\mathcal{S}` representing the likelyhood if the "next state" given the current state and the action
      of the "policy"
     

This tuple
:math:`(s_t, r_t)` is then given to the "agent" / "automaton" / "algorithm" which in turns produce the action :math:`a_{t+1}`

This alternation :math:`\dots \to a \to (s, r) \to a \to \dots` is done for a certain number of "steps" called :math:`T`.

We will call the list :math:`s_{1} \to a_1 \to (s_2, r_2) \to \dots \to a_{T-1} \to (s_{T}, r_T)`
an "episode".



In this section, we will suppose that:

#. there is a "simulator" [informatically, this is the Backend, detailed in the :ref:`backend-module` section of the documentation]
   that is able to compute some informations (*eg* flows on powerlines, active production value of generators etc.)
   from some other information given by the Environment (see :ref:`environment-module` for details about the
   way the `Environment` is coded and refer to :class:`grid2op.Action._backendAction._BackendAction` for list 
   of all available informations informatically available for the solver). 
#. some 

To make a parrallel with some other available environments you can view:

#. The "simulator" represents the physics as in all `"mujoco" environments <https://gymnasium.farama.org/environments/mujoco/>`_ 
   *eg* `Ant <https://gymnasium.farama.org/environments/mujoco/ant>`_ or 
   `Inverted Pendulum <https://gymnasium.farama.org/environments/mujoco/inverted_pendulum>`_ The "simulator" is really the same 
   concept in grid2op and in these environments.
#. 


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


Some constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


.. include:: final.rst
