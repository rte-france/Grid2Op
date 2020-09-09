Runner
===================================

Objectives
-----------
The runner class aims at:

i) facilitate the evaluation of the performance of :class:`grid2op.Agent` by performing automatically the
   "open ai gym loop" (see below)
ii) define a format to store the results of the evaluation of such agent in a standardized manner
iii) this "agent logs" can then be re read by third party applications, such as
     `grid2viz <https://github.com/mjothy/grid2viz>`_ or by internal class to ease the study of the behaviour of
     such agent, for example with the classes :class:`grid2op.Episode.EpisodeData` or
     :class:`grid2op.Episode.EpisodeReplay`
iv) allow easy use of parallelization of this assessment.

Basically, the runner simplifies the assessment of the performance of some agent. This is the "usual" gym code to run
an agent:

.. code-block:: python

    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
    NB_EPISODE = 10  # assess the performance for 10 episodes, for example
    for i in range(NB_EPISODE):
        reward = env.reward_range[0]
        done = False
        obs = env.reset()
        while not done:
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)

The above code does not store anything, cannot be run easily in parallel and is already pretty verbose.
To have a shorter code, that saves most of
the data (and make it easier to integrate it with other applications) we can use the runner the following way:

.. code-block:: python

    import grid2op
    from grid2op.Runner import Runner
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    NB_EPISODE = 10  # assess the performance for 10 episodes, for example
    NB_CORE = 2  # do it on 2 cores, for example
    PATH_SAVE = "agents_log"  # and store the results in the "agents_log" folder
    runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
    runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)

As we can see, with less lines of code, we could execute parallel assessment of our agent, on 10 episode
and save the results (observations, actions, rewards, etc.) into a dedicated folder.

If your agent is inialiazed with a custom `__init__` method that takes more than the action space to be built,
you can also use the Runner pretty easily by passing it an instance of your agent, for example:

.. code-block:: python

    import grid2op
    from grid2op.Runner import Runner
    env = grid2op.make()
    NB_EPISODE = 10  # assess the performance for 10 episodes, for example
    NB_CORE = 2  # do it on 2 cores, for example
    PATH_SAVE = "agents_log"  # and store the results in the "agents_log" folder

    # initilize your agent
    my_agent = FancyAgentWithCustomInitialization(env.action_space,
                                                  env.observation_space,
                                                  "whatever else you want"
                                                  )

    # and proceed as following for the runner
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
    runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)

Other tools are available for this runner class, for example the easy integration of progress bars. See bellow for
more information.

Detailed Documentation by class
-------------------------------
.. automodule:: grid2op.Runner
    :members:
    :private-members:
    :special-members:
    :autosummary:

.. include:: final.rst