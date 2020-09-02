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

Usage
------

In this section we present some way to use the :class:`Environment` class.

Basic Usage
++++++++++++
This example is adapted from gym documentation available at
`gym random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ ):

.. code-block:: python

    import grid2op
    from grid2op.BaseAgent import DoNothingAgent
    env = grid2op.make()
    agent = DoNothingAgent(env.action_space)
    env.seed(0)  # for reproducible experiments
    episode_count = 100  # i want to make 100 episodes

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    for i in range(episode_count):
        ob = env.reset()
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

    # Close the env and write monitor result info to disk
    env.close()
    print("The total reward was {:.2f}".format(total_reward))


What happens here is the following:

- `ob = env.reset()` will reset the environment to be usable again. It will load, by default
  the next "chronics" (you can imagine chronics as the graphics of a video game: it tells where
  the enemies are located, where are the walls, the ground etc. - each chronics can be
  thought a different "game level").
- `action = agent.act(ob, reward, done)` will chose an action facing the observation `ob`.
  This action should be of type :class:`grid2op.Action.BaseAction` (or one of its derivate class).
  In case of a video game that would be you receiving and observation (usually display on the screen)
  and action on a controller. For example you could chose to go "left" / "right" / "up" or "down".
  Of course in the case of the powergrid the actions are more complicated that than.
- `ob, reward, done, info = env.step(action)` is the call to go to the next steps. You can imagine
  it as being a the next "frame". To continue the parallel with video games, at the previous line
  you asked "pacman" to go left (for example) and then the next frame is displayed (here returned
  as an new observation `ob`.

In

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Environment
    :members:
    :autosummary:

.. include:: final.rst