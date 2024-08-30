.. currentmodule:: grid2op.Environment

.. _environment-module:

Environment
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

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
    env = grid2op.make("l2rpn_case14_sandbox")
    agent = RandomAgent(env.action_space)
    env.seed(0)  # for reproducible experiments
    episode_count = 100  # i want to make 100 episodes

    # i initialize some useful variables
    reward = 0
    done = False
    total_reward = 0

    # and now the loop starts
    for i in range(episode_count):
        obs = env.reset()
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

    # Close the env and write monitor result info to disk
    env.close()
    print("The total reward was {:.2f}".format(total_reward))


What happens here is the following:

- `obs = env.reset()` will reset the environment to be usable again. It will load, by default
  the next "chronics" (you can imagine chronics as the graphics of a video game: it tells where
  the enemies are located, where are the walls, the ground etc. - each chronics can be
  thought a different "game level").
- `action = agent.act(obs, reward, done)` will chose an action facing the observation `ob`.
  This action should be of type :class:`grid2op.Action.BaseAction` (or one of its derivate class).
  In case of a video game that would be you receiving and observation (usually display on the screen)
  and action on a controller. For example you could chose to go "left" / "right" / "up" or "down".
  Of course in the case of the powergrid the actions are more complicated that than.
- `obs, reward, done, info = env.step(action)` is the call to go to the next steps. You can imagine
  it as being a the next "frame". To continue the parallel with video games, at the previous line
  you asked "pacman" to go left (for example) and then the next frame is displayed (here returned
  as an new observation `obs`).

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
impact then you might consult the :ref:`environment-module-data-pipeline` page of the doc.

Go to the next scenario
++++++++++++++++++++++++

Starting grid2op 1.9.8 we attempt to make an easier user experience in the
selection of time series, seed, initial state of the grid, etc.

All of the above can be done when calling `env.reset()` function.

For customizing the seed, you can for example do:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)

    obs = env.reset(seed=0)

For customizing the time series id you want to use:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)

    obs = env.reset(options={"time serie id": 1})  # time serie by id (sorted alphabetically)
    # or
    obs = env.reset(options={"time serie id": "0001"})  # time serie by name (folder name)

For customizing the initial state of the grid, for example forcing the
powerline 0 to be disconnected in the initial observation:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)

    init_state_dict = {"set_line_status": [(0, -1)]}
    obs = env.reset(options={"init state": init_state_dict})


Feel free to consult the documentation of the :func:`Environment.reset` function
for more information (this doc might be outdated, the one of the function should 
be more up to date with the code).

.. note::
    In the near future (next few releases) we will also attempt to make the 
    customization of the `parameters` or the `skip number of steps`, `maximum duration 
    of the scenarios` also available in `env.reset()` options.

.. _environment-module-chronics-info:

Time series Customization
++++++++++++++++++++++++++

Study always the same time serie
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    env = grid2op.make("l2rpn_case14_sandbox")
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
        # with recent grid2op 
        obs = env.reset(options={"time serie id": THE_CHRONIC_ID})
        ###################################

        ###################################
        # 'old method (oldest grid2op version)'
        # env.set_id(THE_CHRONIC_ID)
        # obs = env.reset()
        ###################################

        # now play the episode as usual
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

    # Close the env and write monitor result info to disk
    env.close()
    print("The total reward was {:.2f}".format(total_reward))

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Shuffle the chronics order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some other usecase, you might want to go through the whole set of chronics, and then loop again
through them, but in a different order (remember that by default it will always loop in the same
order 0, 1, 2, 3, ..., 0, 1, 2, 3, ..., 0, 1, 2, 3, ...).

Again, doing so with grid2op is rather easy. To that end you can use the `chronics_handler.shuffle` function
that will do exactly that. You can use it like this:

.. code-block:: python

    import numpy as np
    import grid2op
    from grid2op.Agent import RandomAgent
    env = grid2op.make("l2rpn_case14_sandbox")
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

        obs = env.reset()
        # now play the episode as usual
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Skipping some chronics
^^^^^^^^^^^^^^^^^^^^^^
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
    env = grid2op.make("l2rpn_case14_sandbox")
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
        obs = env.reset()
        # now play the episode as usual
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Sampling the chronics
^^^^^^^^^^^^^^^^^^^^^^^

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
    env = grid2op.make("l2rpn_case14_sandbox")
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
        obs = env.reset()

        # now play the episode as usual
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

**NB** here we have a constant vector of probabilities, but you might imagine adapting it during the training,
for example to oversample scenarios your agent is having trouble to solve during the training.

Skipping some time steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    env = grid2op.make("l2rpn_case14_sandbox")
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
        obs = env.reset()

        ###################################
        # below are the two lines added
        env.fast_forward_chronics(42)
        obs = env.get_obs()
        ###################################

        # now play the episode as usual
        while True:
           action = agent.act(obs, reward, done)
           obs, reward, done, info = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

(as always added line compared to the base code are highlighted: they are "circle" with `#####`)

Generating chronics that are always new
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 1.6.6
    This functionality is only available for some environments, for example "l2rpn_wcci_2022"

.. warning::
    A much better alternative to this class is to have a "process" generate the data, thanks to the
    :func:`grid2op.Environment.Environment.generate_data` and then to reload
    the data in a (separate) training script.

    This is explained in section :ref:`generate_data_flow` of the documentation.

Though it is not recommended at all (for performance reasons), you have, starting from grid2op 1.6.6 (and using
a compatible environment *eg* "l2rpn_wcci_2022") to generate a possibly infinite amount of data thanks to the
:class:`grid2op.Chronics.FromChronix2grid` class.

The data generation process is rather slow for different reasons. The main one is that
the data need to meet a lot of "constraints" to be realistic, some of them are
given in the :ref:`modeled-elements-module` module. On our machines, it takes roughly
40-50 seconds to generate a weekly scenario for the `l2rpn_wcci_2022` environment (usually
an agent will fail in 1 or 2s... This is why we do not recommend to use it)

To generate data "on the fly" you simply need to create the environment with the right
chronics class as follow:

.. code-block:: python

        import grid2op
        from grid2op.Chronics import FromChronix2grid
        env_nm = "l2rpn_wcci_2022"  # only compatible environment at time of writing
        
        env = grid2op.make(env_nm,
                           chronics_class=FromChronix2grid,
                           data_feeding_kwargs={"env_path": os.path.join(grid2op.get_current_local_dir(), env_nm),
                                                "with_maintenance": True,  # whether to include maintenance (optional)
                                                "max_iter": 2 * 288,  # duration (in number of steps) of the data generated (optional)
                                                }
                           )


And this is it. Each time you call `env.reset()` it will internally call `chronix2grid` package to generate 
new data for this environment (this is why `env.reset()` will take roughly 50s...).

.. warning::
    For this class to be available, you need to have the "chronix2grid" package installed and working.

    Please install it with `pip intall grid2op[chronix2grid]` and make sure to have the `coinor-cbc` 
    solver available on your system (more information at https://github.com/bdonnot/chronix2grid#installation)

.. warning::
    Because I know from experience warnings are skipped half of the time: **please consult** :ref:`generate_data_flow` **for
    a better way to generate infinite data** !

.. _generate_data_flow:

Generate and use an "infinite" data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 1.6.6


.. warning::
    For this class to be available, you need to have the "chronix2grid" package installed and working.

    Please install it with `pip intall grid2op[chronix2grid]` and make sure to have the `coinor-cbc` 
    solver available on your system (more information at https://github.com/bdonnot/chronix2grid#installation)

In this section we present a new way to generate possibly an infinite amount of data for training your agent (
in case the data shipped with the environment are too limited).

One way to do this is to split the data "generation" process on one python script, and the data "consumption" process
(for example by training an agent) on another one.

This is much more efficient than using the :class:`grid2op.Chronics.FromChronix2grid` because you will not spend 50s
waiting the data to be generated at each call to `env.reset()` after the episode is over.

First, create a script to generate all the data that you want. For example in the script "generation.py":

.. code-block:: python

    import grid2op            
    env_name = "l2rpn_wcci_2022"  # only compatible with what comes next (at time of writing)
    env = grid2op.make(env_name)
    nb_year = 50  # or any "big" number...
    env.generate_data(nb_year=nb_year)  # generates 50 years of data 
    # (takes roughly 50s per week, around 45mins per year, in this case 50 * 45 mins = 37.5 hours)

Then create a script to "consume" your data, for example by training an agent (say "train.py") 
[we demonstrate it with l2rpn baselines but you can use whatever you want]:

.. code-block:: python

    import os
    import grid2op
    from lightsim2grid import LightSimBackend  # highly recommended for speed !
    
    env_name = "l2rpn_wcci_2022"  # only compatible with what comes next (at time of writing)
    env = grid2op.make(env_name, backend=LightSimBackend())
    
    # now train an agent
    # see l2rpn_baselines package for more information, for example
    # l2rpn-baselines.readthedocs.io/
    from l2rpn_baselines.PPO_SB3 import train
    nb_iter = 10000  # train for that many iterations
    agent_name = "WhaetverIWant"  # or any other name
    agent_path = os.path.expand("~")  # or anywhere else on your computer
    trained_agent = train(env,
                          iterations=nb_iter,
                          name=agent_name,
                          save_path=agent_path)
    # this agent will be trained only on the data available at the creation of the environment

    # the training loop will take some time, so more data will be generated when it's over
    # reload them
    env.chronics_handler.init_subpath()
    env.chronics_handler.reset()

    # and retrain your agent including the data you just generated
    trained_agent = train(env,
                          iterations=nb_iter,
                          name=agent_name,
                          save_path=agent_path,
                          load_path=agent_path
                          )

    # once it's over, more time has passed, and more data are available
    # reload them
    env.chronics_handler.init_subpath()
    env.chronics_handler.reset()

    # and retrain your agent
    trained_agent = train(env,
                          iterations=nb_iter,
                          name=agent_name,
                          save_path=agent_path,
                          load_path=agent_path
                          )

    # well you got the idea
    # etc. etc.

.. warning:: 
    This way of doing things will always increase the size of the data in your hard drive.
    We do recommend to somehow delete some of the data from time to time

    Deleting the data you be done before the `env.chronics_handler.init_subpath()` for example:

    .. code-block :: python

        ### delete the folder you want to get rid off
        names_folder_to_delete = ... 
        # To build `names_folder_to_delete`
        # you could for examaple:
        # - remove the `nth` oldest directories
        #   see: https://stackoverflow.com/questions/47739262/find-remove-oldest-file-in-directory
        # - or keep only the `kth`` most recent directories
        # - or keep only `k` folder at random among the one in `grid2op.get_current_local_dir()`
        # - or delete all the oldest files and keep your directory at a fixed size
        #   see: https://gist.github.com/ginz/1ba7de8b911651cfc9c85a82a723f952
        # etc.

        for nm in names_folder_to_delete:
            shutil.rmtree(os.path.join(grid2op.get_current_local_dir(), nm))
        ####
        # reload the remaining data:
        env.chronics_handler.init_subpath()
        env.chronics_handler.reset()

        # continue normally


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
    nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(pct_val=1., pct_test=1.)

    # and now you can use the training set only to train your agent:
    print(f"The name of the training environment is \\"{nm_env_train}\\"")
    print(f"The name of the validation environment is \\"{nm_env_val}\\"")
    print(f"The name of the test environment is \\"{nm_env_test}\\"")
    env_train = grid2op.make(nm_env_train)

You can then use, in the above case:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # matching above

    env_train = grid2op.make(env_name+"_train")  # to only use the "training chronics"
    # do whatever you want with env_train

And then, at time of validation:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # matching above

    env_val = grid2op.make(env_name+"_val") # to only use the "validation chronics"
    # do whatever you want with env_val

    # and of course
    env_test = grid2op.make(env_name+"_test")

Customization
-------------

Environments can be customized in three major ways:

- `Backend`: you change the solver that computes the state of the power more or less faste or be more realistically
- `Parameters`: you change the behaviour of the Environment. For example you can prevent the powerline to be
  disconnected when too much current flows on it etc.
- `Rules`: you can affect the operational constraint that your agent must meet. For example you can affect
  more or less powerlines in the same action etc.

You can do these at creation time:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # or any other name

    # create the regular environment:
    env_reg = grid2op.make(env_name)

    # to change the backend
    # (here using the lightsim2grid faster backend)
    from lightsim2grid import LightSimBackend
    env_faster = grid2op.make(env_name, backend=LightSimBackend())

    # to change the parameters, for example
    # to prevent line disconnect when there is overflow
    param = env_reg.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    env_easier = grid2op.make(env_name, param=param)

Of course you can combine everything. More examples are given in section :ref:`env_cust_makeenv`. 

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Environment
    :members:
    :autosummary:

.. include:: final.rst