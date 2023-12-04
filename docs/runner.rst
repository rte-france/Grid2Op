.. _runner-module:

Runner
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

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
    env = grid2op.make("l2rpn_case14_sandbox")
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
    env = grid2op.make("l2rpn_case14_sandbox")
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
    env = grid2op.make("l2rpn_case14_sandbox")
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

.. _runner-multi-proc-warning:

Note on parallel processing
----------------------------
The "Runner" class allows for parallel execution of the same agent on different scenarios. In this case, each
scenario will be run in independent process.

Depending on the platform and python version, you might end up with some bugs and error like

.. pull-quote::
    AttributeError: Can't get attribute 'ActionSpace_l2rpn_case14_sandbox' on <module 'grid2op.Space.GridObjects'
    from '/lib/python3.8/site-packages/grid2op/Space/GridObjects.py'> Process SpawnPoolWorker-4:

or like:

.. pull-quote::
    File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/multiprocessing/pool.py", line 125, in worker
    result = (True, func(\*args, \*\*kwds))

    File "/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/multiprocessing/pool.py", line 51, in
    starmapstar return list(itertools.starmap(args[0], args[1]))

In this case this means grid2op has a hard time dealing with the multi processing part. In that case, it
is recommended to disable it completely, for example by using, before any call to "runner.run" the following code:

.. code-block:: python

    import os
    from grid2op.Runner import Runner

    os.environ[Runner.FORCE_SEQUENTIAL] = "1"

This will force (starting grid2op >= 1.5) grid2op to use the sequential runner and not deal with the added
complexity of multi processing.

This is especially handy for "windows" system in case of trouble.

For information, as of writing (march 2021):

- macOS with python <= 3.7 will behave like any python version on linux
- windows and macOS with python >=3.8 will behave differently than linux but similarly to one another


Detailed Documentation by class
-------------------------------
.. automodule:: grid2op.Runner
    :members:
    :private-members:
    :special-members:
    :autosummary:

.. include:: final.rst