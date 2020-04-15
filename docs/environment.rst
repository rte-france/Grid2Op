.. currentmodule:: grid2op.Environment

Environment
===================================

Objectives
-----------
This module defines the :class:`Environment` the higher level representation of the world with which an
:class:`grid2op.Agent.BaseAgent` will interact.

The environment receive an :class:`grid2op.Action.BaseAction` from the :class:`grid2op.Agent.BaseAgent` in the
:func:`Environment.step`
and returns an
:class:`grid2op.Observation.BaseObservation` that the :class:`grid2op.Agent.BaseAgent` will use to perform the next action.

An environment is better used inside a :class:`grid2op.Runner.Runner`, mainly because runners abstract the interaction
between environment and agent, and ensure the environment are properly reset after each episode.

It is however totally possible to use as any gym Environment.

Example (adapted from gym documentation available at
`gym random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ ):

.. code-block:: python

    import grid2op
    from grid2op.BaseAgent import DoNothingAgent
    env = grid2op.make()
    agent = DoNothingAgent(env.action_space)
    env.seed(0)
    episode_count = 100
    reward = 0
    done = False
    total_reward = 0
    for i in range(episode_count):
        ob = env.reset()
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, _ = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

    # Close the env and write monitor result info to disk
    env.close()
    print("The total reward was {:.2f}".format(total_reward))


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Environment
    :members:
    :autosummary:

.. include:: final.rst