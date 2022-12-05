.. currentmodule:: grid2op.Rules

Rules of the Game
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

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

Behaviour
----------
In this section, we detail the behaviour of the "cooldown" on the line and substation to make sure  it's clear in the documentation.

The general rules is pretty simple:

1) if `obs.time_before_cooldown_line[l_id] > 0` for a given line id `l_id`, any attempt to modify
   the status of the powerline `l_id` will be illegal.
2) if `obs.time_before_cooldown_line[l_id] == 0` for a given line id `l_id`, it is possible
   to modify the status of powerline `l_id` without any issue. And at the next step, you will 
   need to wait for `env.parameters.NB_TIMESTEP_COOLDOWN_LINE` steps before being able to 
   modify the status of the powerline `l_id`
3) if `obs.time_before_cooldown_sub[s_id] > 0` for a given substation id `s_id`, any attempt to modify
   the topology of the substation `s_id` will be illegal.
4) if `obs.time_before_cooldown_sub[s_id] == 0` for a given substation id `s_id`, it is possible
   to modify the topology of substation `l_id` without any issue. And at the next step, you will 
   need to wait for `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` steps before being able to 
   modify the topology of the substation `l_id`

To illustrate this, let's create an environment that we'll use for an example and force the cooldowns:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)
    env.set_id(0) 
    env.seed(0)

    # for this example, we enforce that we need to wait 3 steps
    # before being able to change again a line or a substation
    param = env.parameters
    param.NB_TIMESTEP_COOLDOWN_SUB = 3
    param.NB_TIMESTEP_COOLDOWN_LINE = 3
    env.change_parameters(param)

    obs = env.reset()

    # in summary (see descriptions bellow for more information) : 
    act_line_1 = env.action_space({"set_line_status": [(1, -1)]})
    obs, reward, done, info = env.step(act_line_1)  # legal, obs.time_before_cooldown_line[1] = 3
    obs, reward, done, info = env.step(act_line_1)  # illegal, obs.time_before_cooldown_line[1] = 2
    obs, reward, done, info = env.step(act_line_1)  # illegal, obs.time_before_cooldown_line[1] = 1
    obs, reward, done, info = env.step(act_line_1)  # illegal, obs.time_before_cooldown_line[1] = 0
    obs, reward, done, info = env.step(act_line_1)  # legal, obs.time_before_cooldown_line[1] = 3

Cooldown on substation
***********************

Now let's do an action on substation 1:

.. code-block:: python

    # let's do an action
    act_sub1 = env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, 1, 1, 1])]}})
    # and see what happens
    obs, reward, done, info = env.step(act_sub1)

Now, you can see that:

.. code-block:: python

    obs.time_before_cooldown_sub
    # >>> array([0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

This is because in the parameters we specified `param.NB_TIMESTEP_COOLDOWN_SUB = 3` and that we 
acted on substaiton 1. In this case, the substation 1 `obs.time_before_cooldown_sub[1]` is 3.

And as long as `obs.time_before_cooldown_sub[1] > 0` you will not be able to act on this
substation again, whether you do the same action or another.

.. code-block:: python

    # let's do another action
    act_sub1_ = env.action_space({"set_bus": {"substations_id": [(1, [1, 2, 2, 1, 2, 1])]}})
    # and see what happens
    obs, reward, done, info = env.step(act_sub1_)
    info['is_illegal']
    # >>> True  => the action is illegal
    info["exception"]
    # >>> [Grid2OpException IllegalAction IllegalAction('Substation with ids [1] have been modified illegally (cooldown)')]
    # the exception raised explains why

    # at this stage we have to wait still 2 steps before doing an action
    obs.time_before_cooldown_sub
    # >>> array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

You can do an action on another substation without any issue:

.. code-block:: python

    # we can do an action on another substation if we want to
    act_sub2 = env.action_space({"set_bus": {"substations_id": [(8, [2, 2, 1, 1, 1])]}})
    obs, reward, done, info = env.step(act_sub2)
    info['is_illegal']
    # >>> False  => the action is perfectly legal
    info["exception"]
    # >>> []
    obs.time_before_cooldown_sub
    # >>> array([0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0], dtype=int32)

At this stage you still cannot do an action on substation 1:

.. code-block:: python

    # we can do an action on another substation if we want to
    obs, reward, done, info = env.step(act_sub1)
    info['is_illegal']
    # >>> True  => the action is illegal
    info["exception"]
    # >>> [Grid2OpException IllegalAction IllegalAction('Substation with ids [1] have been modified illegally (cooldown)')]
    # the exception raised explains why

    obs.time_before_cooldown_sub
    # >>> array([0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], dtype=int32)

And now you can (after 3 steps, and "because" `obs.time_before_cooldown_sub[1] == 0`)
you can do an action on the substation 1, as shown bellow:

.. code-block:: python

    # we can do an action on another substation if we want to
    obs, reward, done, info = env.step(act_sub1_)
    info['is_illegal']
    # >>> False  => the action is perfectly legal, you have waited 3 steps before being able to do an action on sub 1 again
    info["exception"]
    # >>> []

    obs.time_before_cooldown_sub
    # >>> array([0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int32)

Cooldown on powerlines
***********************

Actions on powerline status behaves exactly the same way as actions on substation:

1) if `obs.time_before_cooldown_line[l_id] > 0` for a given line id `l_id`, any attempt to modify
   the status of the powerline `l_id` will be illegal.
2) if `obs.time_before_cooldown_line[l_id] == 0` for a given line id `l_id`, it is possible
   to modify the status of powerline `l_id` without any issue. And at the next step, you will 
   need to wait for `env.parameters.NB_TIMESTEP_COOLDOWN_LINE` steps before being able to 
   modify the status of the powerline `l_id`

TODO explain in more details if needed

Corner cases
***********************

TODO

1) You can modify the status of powerline by modifying the topology !
2) you can change, at any step a powerline and a line status
3) see :ref:`action_powerline_status`


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Rules
    :members:
    :private-members:
    :special-members:
    :autosummary:

.. include:: final.rst
