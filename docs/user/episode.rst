Episode
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
Grid2op defines some special function that help with restoring agent that has run during some episode that has been
saved by the runner.

Here are some basic usage.

First you run an episode:

.. code-block:: python

    import grid2op
    from grid2op.Runner import Runner

    # I create an environment
    env = grid2op.make("rte_case5_example", test=True)

    # I create the runner
    runner = Runner(**env.get_params_for_runner())
    path_save = "/I/SAVED/RESULTS/THERE"

    # I start the runner and save the results in "/I/SAVED/RESULTS/THERE"
    # I start the evaluation on 2 different episode
    res = runner.run(path_save=path_save, nb_episode=2)

Second you can reload the data (here to plot the different productions active values):

.. code-block:: python

    import grid2op
    from grid2op.Episode import EpisodeData

    # I study only the first episode saved, because... why not
    path_saved = "/I/SAVED/RESULTS/THERE"  # same path as before

    li_episode = EpisodeData.list_episode(path_saved)
    full_path, episode_studied = li_episode[0]
    this_episode = EpisodeData.from_disk(path_agent, episode_studied)

    # now the episode is loaded, and you can easily iterate through the observation, the actions etc.
    for act in this_episode.actions:
        print(act)

    for i, obs in enumerate(this_episode.observations):
        print("At step {} the active productions were {}".format(i, obs.prod_p))

    # etc. etc.



Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Episode
    :members:
    :autosummary:

.. include:: final.rst

