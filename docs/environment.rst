.. currentmodule:: grid2op.Environment
.. _environment-module:

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

.. _environment-module-usage:

Usage
------

In this section we present some way to use the :class:`Environment` class.

Basic Usage
++++++++++++
This example is adapted from gym documentation available at
`gym random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ ):

.. code-block:: python

    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
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
  as an new observation `ob`).

You might want to customize this general behaviour in multiple way:

- you might want to study only one chronics (equivalent to only one level of a video game)
  see `Study always the same chronics`_
- you might want to loop through the chronics, but not always in the same order. If that is the case
  you might want to consult the section `Shuffle the chronics order`_
- you might also have spotted some chronics that have bad properties. In this case, you can
  "remove" them from the environment (they will be ignored). This is explained in `Skipping some chronics`_
- you might also want to select at random, the next chronic you will
  use. This allows some compromise between all the above solution. Instead of ignoring some chronics you might want
  to select them less frequently, instead of always using the same one, you can sampling it more often and of
  course, because the sampling is done randomly it's unlikely that the order will remain the same. To use that
  you can check the `Sampling the chronics`_

In a different scenarios, you might also want to skip the first time steps of the chronics, that would
be equivalent to starting into the "middle" of a video game. If that is the case, the subsection
`Skipping some time steps`_ is made for you.

Finally, you might have noticed that each call to "env.reset" might take a while. This can dramatically
increase the training time, especially at the beginning. This is due to the fact that each time
`env.reset` is called, the whole chronics is read from the hard drive. If you want to lower this
impact then you might consult the `Optimize the data pipeline`_ section.

.. _environment-module-chronics-info:

Study always the same chronics
++++++++++++++++++++++++++++++
If you spotted a particularly interesting chronics, or if you want, for some reason
your agent to see only one chronics, you can do this rather easily with grid2op.

All chronics
are given a unique persistent ID (it means that as long as the data is not modified the same
chronics will have always the same ID each time you load the environment). The environment has a
"set_id" method that allows you to use it. Just add "env.set_id(THE\\_ID\\_YOU\\_WANT)" before
the call to "env.reset". This gives the following code:

.. code-block:: python

    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments
    episode_count = 100  # i want to make 100 episodes

    ###################################
    THE_CHRONIC_ID = 42
    ###################################

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    for i in range(episode_count):
        ###################################
        env.set_id(THE_CHRONIC_ID)
        ###################################

        ob = env.reset()

        # now play the episode as usual
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

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Shuffle the chronics order
+++++++++++++++++++++++++++
In some other usecase, you might want to go through the whole set of chronics, and then loop again
through them, but in a different order (remember that by default it will always loop in the same
order 0, 1, 2, 3, ..., 0, 1, 2, 3, ..., 0, 1, 2, 3, ...).

Again, doing so with grid2op is rather easy. To that end you can use the `chronics_handler.shuffle` function
that will do exactly that. You can use it like this:

.. code-block:: python

    import numpy as np
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments
    episode_count = 10000  # i want to make lots of episode

    # total number of episode
    total_episode = len(env.chronics_handler.subpaths)

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    for i in range(episode_count):

        ###################################
        if i % total_episode == 0:
            # I shuffle each time i need to
            env.chronics_handler.shuffle()
        ###################################

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

Skipping some chronics
+++++++++++++++++++++++
Some chronics might be too hard to start a training ("learn to walk before running") and conversely some chronics might
be too easy after a while (you can solve them without doing nothing basically). This is why grid2op allows you to
have some control about which chronics will be used by the environment.

For this purpose you can use the `chronics_handler.set_filter` function. This function takes a
"filtering function" as argument. This "filtering function" takes as argument the full path of the
chronics and should return ``True`` / ``False`` whether or not you want to keep the There is an example:

.. code-block:: python

    import numpy as np
    import re
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments


    ###################################
    # this is the only line of code to add
    # here i select only the chronics that start by "00"
    env.chronics_handler.set_filter(lambda path: re.match(".*00[0-9].*", path) is not None)
    kept = env.chronics_handler.reset()  # if you don't do that it will not have any effect
    print(kept)  # i print the chronics kept
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

Sampling the chronics
+++++++++++++++++++++++

Finally, for even more flexibility, you can choose to sample what will be the next used chronics. To achieve
that you can call the `chronics_handler.sample_next_chronics` This function takes a vector of probabilities
as input (if not provided it assumes all probabilities are equal) and will select an id based on this probability
vector.

In the following example we assume that the vector of probabilities is always the same and that we want, for some
reason oversampling the 10 first chronics, and under sample the last 10:

.. code-block:: python

    import numpy as np
    import re
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments

    episode_count = 10000  # i want to make lots of episode

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    ###################################
    # total number of episode
    total_episode = len(env.chronics_handler.subpaths)
    probas = np.ones(total_episode)
    # oversample the first 10 episode
    probas[:10]*= 5
    # undersample the last 10 episode
    probas[-10:] /= 5
    ###################################

    # and now the loop starts
    # it will only used the chronics selected
    for i in range(episode_count):

        ###################################
        _ = env.chronics_handler.sample_next_chronics(probas)  # this is added
        ###################################
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

**NB** here we have a constant vector of probabilities, but you might imagine adapting it during the training,
for example to oversample scenarios your agent is having trouble to solve during the training.

Skipping some time steps
+++++++++++++++++++++++++

Another way to customize which data your agent will face is to make as if the chronics started at different date
and time. This might be handy in case a scenario is hard at the beginning but less hard at the end, or if you want
your agent to learn to start controlling the grid at any date and time (in grid2op most of the chronics data
provided start at midnight for example).

To achieve this goal, you can use the :func:`BaseEnv.fast_forward_chronics` function. This function skip a given
number of steps. In the following example, we always skip the first 42 time steps before starting the
episode:

.. code-block:: python

    import numpy as np
    import re
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make()
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

        ###################################
        # below are the two lines added
        env.fast_forward_chronics(42)
        ob = env.get_obs()
        ###################################

        # now play the episode as usual
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

.. _environment-module-data-pipeline:

Optimize the data pipeline
++++++++++++++++++++++++++
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
Nothing (see `Basic Usage`_)    low                44.6
set_chunk (see `Chunk size`_)   ultra low          26.8
`MultifolderWithCache`_         high               11.0
==============================  ================  ===================

As you can see, the default usage uses relatively little memory but takes a while to compute (almost 45s to perform
the 100 episode.) On the contrary, the `Chunk size`_ method uses less memory and is about 40% faster. Storing all
data in memory using the `MultifolderWithCache`_ leads to a large memory footprint, but is also significantly
faster. On this benchmark, it is 75% faster (it takes only 25% of the initial time) than the original method.

Chunk size
^^^^^^^^^^^
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
    env = grid2op.make()
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


MultifolderWithCache
^^^^^^^^^^^^^^^^^^^^^
Another way is to use a dedicated class that stores the data in memory. This is particularly useful
to avoid long and inefficient I/O that are replaced by reading the the complete dataset once and store it
into memory.

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

.. _environment-module-train-val-test:

Splitting into raining, validation, test scenarios
---------------------------------------------------
In machine learning the "training / validation / test" framework is particularly usefull to
avoid overfitting and develop models as performant as possible.

Grid2op allows for such usage at the environment level. There is the possibility to "split" an environment
into training / validation and test (*ie* using only some chronics for training, some others for validation
and some others for testing).

This can be done with:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # or any other...
    env = grid2op.make(env_name)

    # extract 1% of the "chronics" to be used in the validation environment. The other 99% will
    # be used for test
    nm_env_train, nm_env_val = env.train_val_split_random(pct_val=1.)

    # and now you can use the training set only to train your agent:
    print(f"The name of the training environment is \\"{nm_env_train}\\"")
    print(f"The name of the validation environment is \\"{nm_env_val}\\"")
    env_train = grid2op.make(nm_env_train)

You can then use, in the above case:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # matching above

    env_train = grid2op.make(env_name+"_train") to only use the "training chronics"
    # do whatever you want with env_train

And then, at time of validation:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # matching above

    env_train = grid2op.make(env_name+"_val") to only use the "training chronics"
    # do whatever you want with env_train


As of now, grid2op do not support "from the API" the possibility to split with convenient
names a environment a second times. If you want to do a "train / validation / test" split we recommend you to:

1. make a training / test split (see below)
2. split again the training set into training / validation (see below)
3. you will have locally an environment named "trainval" on your computer. This directory will not weight
   more than a few kilobytes.

The example, not really convenient at the moment, please find a feature request if that is a problem for
you:

.. code-block:: python

    import grid2op
    import os

    env_name = "l2rpn_case14_sandbox"  # or any other...
    env = grid2op.make(env_name)

    # retrieve the names of the chronics:
    full_path_data = env.chronics_handler.subpaths
    chron_names = [os.path.split(el)[-1] for el in full_path_data]

    # splitting into training / test, keeping the "last" 10 chronics to the test set
    nm_env_trainval, nm_env_test = env.train_val_split(val_scen_id=chron_names[-10:],
                                                       add_for_val="test",
                                                       add_for_train="trainval")

    # now splitting again the training set into training and validation, keeping the last 10 chronics
    # of this environment for validation
    env_trainval = grid2op.make(nm_env_trainval)  # create the "trainval" environment
    full_path_data = env_trainval.chronics_handler.subpaths
    chron_names = [os.path.split(el)[-1] for el in full_path_data]
    nm_env_train, nm_env_val = env_trainval.train_val_split(val_scen_id=chron_names[-10:],
                                                            remove_from_name="_trainval$")

And later on, you can do, if you followed the names above:

.. code-block:: python

    import grid2op
    import os

    env_name = "l2rpn_case14_sandbox"  # or any other...
    env_train = grid2op.make(env_name+"_train")
    env_val = grid2op.make(env_name+"_val")
    env_test = grid2op.make(env_name+"_test")

And you can also, if you want, delete the folder "l2rpn_case14_sandbox_trainval" from your machine:

.. code-block:: python

    import grid2op
    import os

    env_name = "l2rpn_case14_sandbox"  # or any other...
    env_trainval = grid2op.make(env_name+"_trainval")
    print(f"You can safely delete, if you want, the folder: \n\t\"{env_trainval.get_path_env()}\" \nnow useless.")


Customization
-------------

Environments can be customized in three major ways:

- `Backend`: you change the solver that computes the state of the power more or less faste or be more realistically
- `Parameters`: you change the behaviour of the Environment. For example you can prevent the powerline to be
  disconnected when too much current flows on it etc.
- `Rules`: you can affect the operational constraint that your agent must meet. For example you can affect
  more or less powerlines in the same action etc.

TODO

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Environment
    :members:
    :autosummary:

.. include:: final.rst