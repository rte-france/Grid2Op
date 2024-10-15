.. currentmodule:: grid2op.simulator
.. _simulator_page:

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

It behaves similarly to `env.step(...)` or `obs.simulate(...)` with a few key differences:
    
    - you can "chain" the call to simulator: `simulator.predict(...).predict(...).predict(...)`
    - it does not take into account the "time": no cooldown on lines nor substation, storage
      "state of charge" (energy) does not decrease when you use them
    - no automatic line disconnection: lines are not disconnected when they are above their limit
    - no opponent will act on the grid

Usage
------

You can use it to assess if the grid state is "resilient enough" if the loads and generators increase of 5% :

.. code-block:: python

    import grid2op
    env_name = ...  # any environment name available (eg. "l2rpn_case14_sandbox")
    env = grid2op.make(env_name)

    obs = env.reset()

    #### later in the code, for example in an Agent:

    simulator = obs.get_simulator()

    load_p_stressed = obs.load_p * 1.05
    gen_p_stressed = obs.gen_p * 1.05
    do_nothing = env.action_space()
    simulator_stressed = simulator.predict(act=do_nothing,
                                           new_gen_p=gen_p_stressed,
                                           new_load_p=load_p_stressed)
    if not simulator_stressed.converged:
        # the solver fails to find a solution for this action
        # you are likely to run into trouble if you use that...
        ...  # do something
    obs_stressed = simulator_stressed.current_obs

You can also "chain" the call to simulators, usefull for "model based" strategies

.. code-block:: python

    import grid2op
    env_name = ...  # any environment name available (eg. "l2rpn_case14_sandbox")
    env = grid2op.make(env_name)

    obs = env.reset()

    #### later in the code, for example in an Agent:

    simulator = obs.get_simulator()

    load_p_stressed = obs.load_p * 1.05
    gen_p_stressed = obs.gen_p * 1.05
    do_nothing = env.action_space()
    simulator_stressed = simulator.predict(act=do_nothing, new_gen_p=gen_p_stressed, new_load_p=load_p_stressed)
    if not simulator_stressed.converged:
        # the solver fails to find a solution for this action
        # you are likely to run into trouble if you use that...
        ...  # do something

    act1 = ...
    simulator_afteract1 = simulator_stressed.predict(act=act1)
    act2 = ...
    simulator_afteract2 = simulator_stressed.predict(act=act2)
    # etc.

Another use (though you might need to use dedicated tools for such purpose such as 
https://lightsim2grid.readthedocs.io/en/latest/security_analysis.html ) is to check whether your grid is "N-1 secure" which means
that if a powerline were to be disconnected (for example by the opponent) then the state it is in would not
be too dangerous:

.. code-block:: python

    import numpy as np
    import grid2op
    env_name = ...  # any environment name available (eg. "l2rpn_case14_sandbox")
    env = grid2op.make(env_name)

    obs = env.reset()

    #### later in the code, for example in an Agent:
    simulator = obs.get_simulator()
    act1 = ...  # the action your agent has selected
    simulator_afteract1 = simulator.predict(act=act1)

    for line_id in range(obs.n_line):
        act_disco_line = env.action_space()
        act_disco_line.set_line_status = [(line_id, -1)]
        sim_after_disco_line = simulator_afteract1.predict(act_disco_line)
        if sim_after_disco_line.converged:
            obs_after_disco = sim_after_disco_line.current_obs
            # and for example check the current on all line is bellow the limit
            if np.any(obs_after_disco.rho > 1.):
                # your action does not make the grid "N-1" safe,
                # because if you disconnect line `line_id` you got some overflow
                ...  # do something
        else:
            # the simulator has not found any solution, it is likely that this will lead
            # to a game over in "real time"
            # your grid is not "N-1" safe because disconnection of line `line_id` lead
            # to a non feasible grid state
                ...  # do something

.. warning::
    This module is still under development, if you need any functionality, let us know with a github "feature request" 
    (https://github.com/Grid2Op/grid2op/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=)


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.simulator
    :members:
    :show-inheritance:
    :special-members:
    :autosummary:

.. include:: final.rst
