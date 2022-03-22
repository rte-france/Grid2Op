.. currentmodule:: grid2op.simulator

Simulator
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
The "Simulator" class is used to interact, with a grid2op "api-like" (the concept of `action` and `observation`) with a powergrid simulator that
is independant of the one used by the environment.

This can for example be used for "model based reinforcement learning" or to assess the "validity" of an action in some "more extreme" conditions.

Usage
------

You can use it to assess if the grid state is "resilient enough" if the loads and generators increase of 5% :

.. code-block:: python

    import grid2op
    env_name = ...  # any environment name available
    env = grid2op.make(env_name)

    obs = env.reset()

    #### later in the code, for example in an Agent:

    simulator = obs.get_simulator()

    load_p_stressed = obs.load_p * 1.05
    gen_p_stressed = obs.gen_p * 1.05
    do_nothing = env.action_space()
    simulator_stressed = simulator.predict(act=do_nothing, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed)
    obs_stressed = simulator_stressed.current_obs

You can also "chain" the call to simulators, usefull for "model based" strategies

.. code-block:: python

    import grid2op
    env_name = ...  # any environment name available
    env = grid2op.make(env_name)

    obs = env.reset()

    #### later in the code, for example in an Agent:

    simulator = obs.get_simulator()

    load_p_stressed = obs.load_p * 1.05
    gen_p_stressed = obs.gen_p * 1.05
    do_nothing = env.action_space()
    simulator_stressed = simulator.predict(act=do_nothing, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed)

    act1 = ...
    simulator_afteract1 = simulator_stressed.predict(act=act1)
    act2 = ...
    simulator_afteract2 = simulator_stressed.predict(act=act2)
    # etc.

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.simulator
    :members:
    :show-inheritance:
    :special-members:
    :autosummary:

.. include:: final.rst
