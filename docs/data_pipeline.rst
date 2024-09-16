.. _environment-module-data-pipeline:

Optimize the data pipeline
============================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
--------------------------

Optimizing the data pipeline can be crucial if you want to learn fast, especially at the beginning of the training.
There exists multiple way to perform this task.

First, let's start with a summary of the timing. For this test, i ran, on my personal computer, the following
code to compare the different method.

.. code-block:: python

    import time
    import grid2op
    from grid2op.Chronics import MultifolderWithCache


    ##############################
    # this part changes depending on the method
    env = grid2op.make("l2rpn_neurips_2020_track1_small")
    env.chronics_handler.set_filter(lambda path: re.match(".*37.*", path) is not None)
    kept = env.chronics_handler.reset()  # if you don't do that it will not have any effect
    ##############################

    episode_count = 100
    reward = 0
    done = False
    total_reward = 0

    # only the time of the following loop is measured
    %%time
    for i in range(episode_count):
        ob = env.reset()
        if i % 10 == 0:
            print("10 more")
        while True:
            action = env.action_space.sample()
            ob, reward, done, info = env.step(action)
            total_reward += reward
            if done:
               # in this case the episode is over
               break

Results are reported in the table below:

==============================  ================  ===================
Method used                     memory footprint  time to perform (s)
==============================  ================  ===================
Nothing (see Basic Usage )       low                44.6
set_chunk (see `Chunk size`_ )   ultra low          26.8
`MultifolderWithCache`_          high               11.0
==============================  ================  ===================

As you can see, the default usage uses relatively little memory but takes a while to compute (almost 45s to perform
the 100 episode.) On the contrary, the `Chunk size`_ method uses less memory and is about 40% faster. Storing all
data in memory using the `MultifolderWithCache`_ leads to a large memory footprint, but is also significantly
faster. On this benchmark, it is 75% faster (it takes only 25% of the initial time) than the original method.

Chunk size
+++++++++++
The first think you can do, without changing anything to the code, is to ask grid2op to read the input grid data
by "chunk". This means that, when you call "env.reset" instead of reading all the data representing a full month,
you will read only a subset of it, thus speeding up the IO time by a large amount. In the following example we
read data by "chunk" of 100 (if you want hard drive is accessed to read data 100 time steps by 100 time steps
(instead of reading the full dataset at once) Note that this "technique" can also be used to reduce the memory
footprint (less RAM taken).

.. code-block:: python

    import numpy as np
    import re
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make("l2rpn_case14_sandbox")
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments

    ###################################
    env.chronics_handler.set_chunk_size(100)
    ###################################

    episode_count = 10000  # i want to make lots of episode

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    # it will only used the chronics selected
    for i in range(episode_count):
        ob = env.reset()

        # now play the episode as usual
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

.. note::
    Not all "environment" supports "chunk size". For example if data are generated "on the fly", for now
    you are forced to generate an entire episode, you cannot generate it "piece by piece".

MultifolderWithCache
+++++++++++++++++++++
Another way is to use a dedicated class that stores the data in memory. This is particularly useful
to avoid long and inefficient I/O that are replaced by reading the the complete dataset once and store it
into memory.

.. seealso::
    The documentation of :class:`grid2op.Chronics.Chronics.MultifolderWithCache` for a more
    detailed documentation.

.. versionchanged:: 1.9.0
    Any call to "env.reset()" or "env.step()" without a previous call to `env.chronics_handler.real_data.reset()`
    will raise an error preventing any use of the environment.
    (It is no longer assumed people read, at least partially the documentation.)

.. danger::
    When you create an environment with this chronics class (*eg* by doing 
    `env = make(...,chronics_class=MultifolderWithCache)`), the "cache" is not
    pre loaded, only the first scenario is loaded in memory (to save loading time).
    
    In order to load everything, you NEED to call `env.chronics_handler.reset()`, which,
    by default, will load every scenario into memory. If you want to filter some
    data, for example by reading only the scenario of decembre, you can use the 
    `set_filter` method.
    
    A typical workflow (at the start of your program) when using this class is then:
    
    1) create the environment: `env = make(...,chronics_class=MultifolderWithCache)`
    2) (optional but recommended) select some scenarios: 
       `env.chronics_handler.real_data.set_filter(lambda x: re.match(".*december.*", x) is not None)` 
    3) load the data in memory: `env.chronics_handler.reset()` (see *eg* :func:`grid2op.Chronics.MultifolderWithCache.reset`)
    4) do whatever you want using `env`


This can be achieved with:

.. code-block:: python

    import numpy as np
    import re
    import grid2op
    from grid2op.Agent import RandomAgent
    from grid2op.Chronics import MultifolderWithCache

    ###################################
    env = grid2op.make(chronics_class=MultifolderWithCache)
    # I select only part of the data, it's unlikely the whole dataset can fit into memory...
    env.chronics_handler.set_filter(lambda path: re.match(".*00[0-9].*", path) is not None)
    # you need to do that
    kept = env.chronics_handler.real_data.reset()
    ###################################

    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments

    episode_count = 10000  # i want to make lots of episode

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    # it will only used the chronics selected
    for i in range(episode_count):
        ob = env.reset()

        # now play the episode as usual
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Note that by default the `MultifolderWithCache` class will only load the **first** chronics it sees. You need
to filter it and call `env.chronics_handler.real_data.reset()` for it to work properly.
