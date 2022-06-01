
.. |l2rpn_case14_sandbox_layout| image:: ./img/l2rpn_case14_sandbox_layout.png
.. |R2_full_grid| image:: ./img/R2_full_grid.png
.. |l2rpn_neurips_2020_track1_layout| image:: ./img/l2rpn_neurips_2020_track1_layout.png
.. |l2rpn_neurips_2020_track2_layout| image:: ./img/l2rpn_neurips_2020_track2_layout.png
.. |l2rpn_wcci_2022_layout| image:: ./img/l2rpn_wcci_2022_layout.png


Available environments
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Content of an environment
---------------------------

A grid2op "environment" is represented as a folder on your computer. There is one folder for each environment.

Inside each folder / environment there are a few files (as of writing):

- "**grid.json**" (a file): it is the file that describe the powergrid and that can be read by the default backend.
  It is today
  mandatory, but we could imagine a file in a different format. Note that in this case,
  this environment will not be compatible with the default backend.
- "**config.py**" (a file): this file is imported when the environment is loaded. It is used to parametrize the way
  the environment is made. It should define a "config" variable. This "config" is dictionary that is used to initialize
  the environment. They key should be variable names. See example of such "*config.py*" file in provided environment.

It can of course contain other information, among them:

- "**chronics**" (a folder) [recommended]: this folder contains the information to generate the production / loads at each steps.
  It can
  itself contain multiple folder, depending on the :class:`grid2op.Chronics.GridValue` class used. In most available
  environment, the class :class:`grid2op.Chronics.Multifolder` is used. This folder is optional, though it is present
  in most grid2op environment provided by default.
- "**grid_layout.json**" (a file) [recommended]: gives, for each substation its coordinate *(x,y)* when plotted. It is optional, but
  we
  strongly encourage to have such. Otherwise, some tools might not work (including all the tool to represent it, such
  as the renderer (`env.render`), the `EpisodeReplay` or even some other dependency package, such as Grid2Viz).
- "**prods_charac.csv**" (file): [see :func:`grid2op.Backend.Backend.load_redispacthing_data` for a
  description of this file]
  This contains all the information related to "ramps", "pmin / pmax", etc. This file is optional (grid2op can
  perfectly run without it). However, if absent, then the classes
  :attr:`grid2op.Space.GridObjects.redispatching_unit_commitment_availble` will be set to ``False`` thus preventing
  the use of some feature that requires it (for example *redispatching* or *curtailment*)
- "**storage_units_charac.csv**" (file): [see :func:`grid2op.Backend.Backend.load_storage_data` for a description
  of this file]
  This file is used for a description of the storage units. It is a description of the storage units needed by grid2op.
  This is optional if you don't have any storage units on the grid but required if there are (otherwise a
  `BackendError` will be raised).
- "**difficulty_levels.json**" (file): This file is useful is you want to define different "difficulty" for your
  environment. It should be a valid json with keys being difficulty levels ("0" for easiest to "1", "2", "3", "4", "5"
  , ..., "10", ..., "100", ... or "competition" for the hardest / closest to reality difficulty).

And this is it for default environment.

You can highly customize everything. Only the "config.py" file is really mandatory:

- if you don't care about your environment to run on the default "Backend", you can get rid of the "grid.json"
  file. In that case you will have to use the "keyword argument" "backend=..." when you create your environment
  (*e.g* `env = grid2op.make(..., backend=...)` ) This is totally possible with grid2op and causes absolutely
  no issues.
- if you code another :class:`grid2op.Chronics.GridValue` class, you can totally get rid of the "chronics" repository
  if you want to. In that case, you will need to either provide "chronics_class=..." in the config.py file,
  or initialize with `env = grid2op.make(..., chronics_class=...)`
- if your grid data format contains enough information for grid2op to initialize the redispatching and / or storage
  data then you can freely use it and override the :func:`grid2op.Backend.Backend.load_redispacthing_data` or
  :func:`grid2op.Backend.Backend.load_storage_data` and read if from the grid file without any issues at all.

List of available environment
------------------------------

How to get the up to date list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete list of **test** environments can be found using:

.. code-block:: python

    import grid2op
    grid2op.list_available_test_env()

And the list of environment that can be downloaded is given by:

.. code-block:: python

    import grid2op
    grid2op.list_available_remote_env()

In this case, remember that the data will be downloaded in:

.. code-block:: python

    import grid2op
    grid2op.get_current_local_dir()

Description of some environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The provided list has been updated early April 2021:

================================  ===========  =============  ==========  ===============  ============================
env name                          grid size     maintenance    opponent    redisp.           storage unit
================================  ===========  =============  ==========  ===============  ============================
:ref:`l2rpn_case14_sandbox`        14 sub.       ❌               ❌  ️         ✔️ ️                 ❌
:ref:`l2rpn_wcci_2020`             36 sub.       ✔️  ️         ❌  ️         ✔️ ️                ❌
:ref:`l2rpn_neurips_2020_track1`   36 sub.       ✔️  ️         ✔️ ️       ✔️ ️                 ❌
:ref:`l2rpn_neurips_2020_track2`   118 sub.      ✔️  ️         ❌   ️         ✔️ ️                 ❌
:ref:`l2rpn_icaps_2021`            36 sub.       ✔️  ️         ✔️ ️       ✔️ ️                 ❌
:ref:`l2rpn_wcci_2022`             118 sub.      ✔️  ️         ✔️ ️       ✔️ ️                 ✔️ ️
\* educ_case14_redisp \*           14 sub.       ❌️             ❌  ️ ️       ✔️ ️                 ❌
\* educ_case14_storage \*          14 sub.       ❌️             ❌   ️         ✔️ ️                 ✔️
\* rte_case5_example \*            5 sub.        ❌️             ❌  ️ ️        ❌ ️ ️                  ❌
\* rte_case14_opponent \*          14 sub.       ❌️             ✔️ ️        ❌ ️ ️                  ❌
\* rte_case14_realistic \*         14 sub.       ❌️             ❌ ️  ️        ✔️      ️             ❌
\* rte_case14_redisp \*            14 sub.       ❌️             ❌ ️  ️        ✔️ ️                  ❌
\* rte_case14_test \*              14 sub.       ❌️             ❌ ️  ️        ❌ ️ ️                  ❌
\* rte_case118_example \*          118 sub.      ❌️             ❌   ️         ✔️ ️                  ❌
================================  ===========  =============  ==========  ===============  ============================

To create regular environment, you can do:

.. code-block:: python

    import grid2op
    env_name = ... # for example "educ_case14_redisp" or "l2rpn_wcci_2020"
    env = grid2op.make(env_name)

The first time an environment is called, the data for this environment will be downloaded from the internet. Make sure
to have an internet connection where you can access https website (such as https://github.com ). Afterwards, the data
are stored on your computer and you won't need to download it again.

.. warning::

    Some environment have different names. The only difference in this case will be the suffixes "_large" or "_small"
    appended to them.

    This is because we release different version of them. The "basic" version are for testing purpose,
    the "_small" are for making standard experiment. This should be enough with most use-case including training RL
    agent.

    And you have some "_large" dataset for larger studies. The use of "large" dataset is not recommended. It can create
    way more problem than it solves (for example, you can fit a small dataset entirely in memory of
    most computers, and having that, you can benefit from better performances - your agent will be able to perform
    more steps per seconds. See :ref:`environment-module-data-pipeline` for more information).
    These datasets were released to address some really specific use in case were "overfitting" were encounter, we are
    still unsure about their usefulness even in this case.

    This is the case for "l2rpn_neurips_2020_track1" and "l2rpn_neurips_2020_track2". To create them, you need to do
    `env = grid2op.make("l2rpn_neurips_2020_track1_small")` or `env = grid2op.make("l2rpn_neurips_2020_track2_small")`

So to create both the environment, we recommend:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_neurips_2020_track1_small"  # or "l2rpn_neurips_2020_track2_small"
    env = grid2op.make(env_name)

.. warning::

    Environment with \* are reserved for testing / education purpose only. We do not recommend to perform
    extensive studies with them as they contain only little data.

For these testing environments (the one with \* around them in the above list):

.. code-block:: python

    import grid2op
    env_name = ... # for example "l2rpn_case14_sandbox" or "educ_case14_storage"
    env = grid2op.make(env_name, test=True)

.. note::

    More information about each environment is provided in each of the sub section below
    (one sub section per environment)

.. _l2rpn_case14_sandbox:

l2rpn_case14_sandbox
+++++++++++++++++++++

This dataset uses the IEEE case14 powergrid slightly modified (a few generators have been added).

It counts 14 substations, 20 lines, 6 generators and 11 loads. It does not count any storage unit.

We recommend to use this dataset when you want to get familiar with grid2op, with powergrid modeling  or RL. It is a
rather small environment where you can understand and actually see what is happening.

This grid looks like:

|l2rpn_case14_sandbox_layout|


.. _l2rpn_wcci_2022_dev:

l2rpn_wcci_2022
++++++++++++++++

This environment will come in two "variations":

- `l2rpn_wcci_2022_dev`: development version (might not be totally finished at time of writing), to be used for
  test only, only a few snapshots are available.
- `l2rpn_wcci_2022` : (equivalent of 32 years of powergrid data at 5 mins interval) weights ~1.7 GB

You have the possibility, provided that you installed `chronix2grid` (with `pip install grid2op[chronix2grid]`), to generate as
much data as you want with the :func:`grid2op.Environment.Environment.generate_data` function. See its documentation for more information.

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_wcci_2022"
    env = grid2op.make(env_name)

It counts 118 substations, 186 powerlines, 91 loads and 62 loads. It will be used for the L2RPN competitions at WCCI in 2022.

|l2rpn_wcci_2022_layout|

You can add as many chronics as you want to this environment with the code:

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_wcci_2022"
    env = grid2op.make(env_name)

    nb_year = 1 # or any postive integer
    env.generate_data(nb_year=nb_year)

It might take a while (so we advise you to get a nice cup of tea, coffee or anything)
and will only work if you installed chronix2grid package.


.. _l2rpn_icaps_2021:

l2rpn_icaps_2021
++++++++++++++++

This environment comes in 3 different "variations" (depending on the number of chronics available):

- `l2rpn_icaps_2021_small` (1 GB equivalent of 50 years of powergrid data at 5 mins interval,
  so `4 838 400` different steps !)
- `l2rpn_icaps_2021_large` (4.8 GB equivalent of ~250 years of powergrid data at 5 mins interval,
  so `23 804 928` different steps !)
- `l2rpn_icaps_2021` (use it for test only, only a few snapshots are available)

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_icaps_2021_small"
    env = grid2op.make(env_name)

It is based on the same powergrid as the :ref:`l2rpn_neurips_2020_track1` environment
and was used for the L2RPN ICAPS 2021 competition. It counts 36 substations, 59
powerlines, 22 generators and 37 loads (some of which represents interconnection with 
another grid).

|l2rpn_neurips_2020_track1_layout|


.. _l2rpn_neurips_2020_track1:

l2rpn_neurips_2020_track1
+++++++++++++++++++++++++++

This environment comes in 3 different "variations" (depending on the number of chronics available):

- `l2rpn_neurips_2020_track1_small` (900 MB, equivalent of 48 years of powergrid data at 5 mins interval,
  so `4 644 864` different steps !)
- `l2rpn_neurips_2020_track1_large` (4.5 GB, equivalent of 240 years of powergrid data at 5 mins interval,
  so `23 22 4320` different steps.)
- `l2rpn_neurips_2020_track1` (use it for test only, only a few snapshots are available)

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_neurips_2020_track1_small"
    env = grid2op.make(env_name)

It was the environment used as a training set of the neurips 2020 "L2RPN" competition, for the "robustness" track,
see https://competitions.codalab.org/competitions/25426 .

This environment is part of the IEEE 118 grid, where some generators have been added. It counts 36 substations, 59
powerlines, 22 generators and 37 loads (some of which represents interconnection with 
another grid). The grid is represented in the figure below:

|l2rpn_neurips_2020_track1_layout|

One of the specificity of this grid is that it is actually a subset of a bigger grid. Actually, it represents the grid
"circled" in red in the figure below:

|R2_full_grid|

This explains why there can be some "negative loads" in this environment. Indeed, this loads represent interconnection
with other part of the original grid (emphasize in green in the figure above).


.. _l2rpn_neurips_2020_track2:

l2rpn_neurips_2020_track2
+++++++++++++++++++++++++++

- `l2rpn_neurips_2020_track2_small` (2.5 GB, split into 5 different sub-environment - each being generated from
  slightly different distribution - with 10 years for each sub-environment. This makes, for each sub-environment
  `1 051 200` steps, so `5 256 000` different steps in total)
- `l2rpn_neurips_2020_track2_large` (12 GB, again split into 5 different sub-environment. It is 5 times as large
  as the "small" one. So it counts `26 280 000` different steps. Each containing all the information of all productions
  and all loads. This is a lot of data)
- `l2rpn_neurips_2020_track2` (use it for test only, only a few snapshots are available)

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_neurips_2020_track2_small"
    env = grid2op.make(env_name)

It was the environment used as a training set of the neurips 2020 "L2RPN" competition, for the "robustness" track,
see https://competitions.codalab.org/competitions/25427 .

This environment is the IEEE 118 grid, where some generators have been added. It counts 118 substations, 186
powerlines, 62 generators and 99 loads. The grid is represented in the figure below:

|l2rpn_neurips_2020_track2_layout|

This grid is, as specified in the previous paragraph, a "super set" of the grid used in the other track. It does not
count any "interconnection" with other types of grid.

.. _l2rpn_wcci_2020:

l2rpn_wcci_2020
+++++++++++++++++++++++++++

This environment `l2rpn_wcci_2020`  weight 4.5 GB, representing 240 equivalent years of data at 5 mins resolution, so
`25 228 800` different steps. Unfortunately, you can only download the full dataset.

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "l2rpn_wcci_2020"
    env = grid2op.make(env_name)

It was the environment used as a training set of the WII 2020 "L2RPN" competition
see https://competitions.codalab.org/competitions/24902 .

This environment is part of the IEEE 118 grid, where some generators have been added. It counts 36 substations, 59
powerlines, 22 generators and 37 loads. The grid is represented in the figure below:

|l2rpn_neurips_2020_track1_layout|

.. note::

    It is an earlier version than the `l2rpn_neurips_2020_track1`. In the `l2rpn_wcci_2020` it is not easy
    to identify which loads are "real" loads, and which are "interconnection" for example.

    Also, the names of some elements (substations, loads, lines, or generators) are different.
    In the `l2rpn_neurips_2020_track1` the names match the one in `l2rpn_neurips_2020_track2` which is not
    the case in `l2rpn_wcci_2020` which make it less obvious that is a subgrid of the IEEE 118.


educ_case14_redisp (test only)
+++++++++++++++++++++++++++++++

It is the same kind of data as the "l2rpn_case14_sandbox" (see above). It counts simply less data and allows
less different type of actions for easier "access". It do not require to dive deep into grid2op to use this environment.

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "educ_case14_redisp"
    env = grid2op.make(env_name, test=True)


educ_case14_storage (test only)
++++++++++++++++++++++++++++++++

Uses the same type of actions as the grid above ("educ_case14_redisp") but counts 2 storage units. The grid on which
it is based is also the IEEE case 14 but with 2 additional storage unit.

We recommend to create this environment with:

.. code-block:: python

    import grid2op
    env_name  = "educ_case14_storage"
    env = grid2op.make(env_name, test=True)

rte_case5_example (test only)
+++++++++++++++++++++++++++++

.. warning::

    We dont' recommend to create this environment at all, unles you want to perform some specific dedicated tests.

A custom made environment, totally fictive, not representative of anything, mainly develop for internal tests and
for super easy representation.

The grid on which it is based has absolutely no "good properties" and is "mainly random" and is not calibrated
to be representative of anything, especially not of a real powergrid. Use at your own risk.


other environments (test only)
++++++++++++++++++++++++++++++++

Some other test environments are available:

- "rte_case14_realistic"
- "rte_case14_redisp"
- "rte_case14_test"
- "rte_case118_example"

.. warning::

    We don't recommend to create any of these environments at all,
    unless you want to perform some specific dedicated tests.

    This is why we don't detail them in this documentation.


Miscellaneous
--------------

Possible workflow to create an environment from existing chronics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this subsection, we will give an example on how to set up an environment in grid2op if you already
have some data that represents loads and productions at each steps. This paragraph aims at making more concrete
the description of the environment shown previously.

For this, we suppose that you already have:
- a powergrid in any type of format that represents the grid you have studied.
- some injections data, in any format (csv, mysql, json, etc. etc.)

The process to make this a grid2op environment is the following:

1) :ref:`create_folder`: create the folder
2) :ref:`grid_json_ex`: convert the grid file / make sure you have a "backend that can read it"
3) :ref:`chronics_folder_ex`: convert your data / make sure to have a "GridValue" that understands it
4) :ref:`config_py_ex`: create the `config.py` file
5) [optional] :ref:`grid_layout_ex`: generate the `grid_layout.json`
6) [optional] :ref:`prod_charac_ex`: generate the `prod_charac.csv`and `storage_units_charac.csv` if needed
7) :ref:`test_env_ex`: charge the environment and test it
8) [optional] :ref:`calibrate_th_lim_ex`: calibrate the thermal limit and set them in the `config.py` file

Each task is briefly described in a following paragraph.

.. _create_folder:

Creating the folder
+++++++++++++++++++++
First you need to create the folder that will represent your environment. Just create an empty folder anywhere
on your computer.

For the sake of the example, we assume here the folder is `EXAMPLE_FOLDER=C:\\Users\\Me\\Documents\\my_grid2op_env`, it
can also be `EXAMPLE_FOLDER=/home/Me/Documents/my_grid2op_env` or
`EXAMPLE_FOLDER=/home/Me/Documents/anything_i_want_really` it does not matter.

.. _grid_json_ex:

Generate the "grid.json" file
+++++++++++++++++++++++++++++

.. note::

    The title of this section is "grid.json" for simplicity. We would like to recall that grid2op do not care about the
    the format used to represent powergrid. It could be an xml, excel, sql, or any format you want, really.

We supposed for this section that you add a file representing a grid at your disposal. So it's time to use it.

From there, there are 3 different situations you can be in:

1) you have a grid in a given format (for example `json` format) and already have at your disposal a type of grid2op
   backend (for example `PandaPowerBackend`) then you don't need to do anything in particular.
2) you have a grid in a given format (for example `.example`) and knows how to convert it to a format for which
   you have a backend (typically: `PandapPowerBackend`, that reads pandapower json file). In that case, you convert the
   grid and you put the converted grid in the directory and you are good. For converters to pandapower, you can
   consult the official pandapower documentation at https://pandapower.readthedocs.io/en/v2.6.0/converter.html .
3) you have a grid in a given format, but don't know how to convert it to a format where you have a backend. In that
   case it might require a bit more work (see details below)

.. note::

   Case 2 above includes the case where you can convert your file in a format not compatible with default
   PandapowerBackend. For example, you could have a grid in sql database, that you know how to convert to a "xml" file
   and you already coded "CustomBackend" that is able to work with this xml file. This is totally fine too !

In all cases, after you converted your file, name it `grid.something` (for example `grid.json` if your grid is
compatible with pandapowerr backend) into the folder `EXAMPLE_FOLDER` (for example
`C:\\Users\\Me\\Documents\\my_grid2op_env`)

The rest of this section is only relevant if you are in case 3 above. You can go to the next section
:ref:`chronics_folder_ex` if you are in case 1 or 2 below.

You have in that two solutions:

1) if you have lots such "conversion in grid2op env to do" or if you think it makes sense for you simulator to
   be used as a grid2op backend outside of your use case, then it's totally worth it to try to create a dedicated
   backend class for your powerflow solver. Once done, you can reuse it or even make it available for other to use it.
2) if you are trying to do a "one shot" things the easiest road would be to try to convert your grid into a format
   that pandapower is able to understand. Pandpower does understand the Matpower format which is pretty common. You
   might check if your grid format is convertible into mapower format, and then convert the matpower format to
   pandapower one (for example). The main point is: try to convert the grid to a format that can be processed by
   the default grid2op backend.

.. _chronics_folder_ex:

Organize the "chronics" folder
+++++++++++++++++++++++++++++++

In this step, you are suppose to provide a way for grid2op to set the value of each production and load at each step.

The first step is then to create a folder named "chronics" in `EXAMPLE_FOLDER` (remember, in our example
`EXAMPLE_FOLDER` was `C:\\Users\\Me\\Documents\\my_grid2op_env`, so you need to create
`C:\\Users\\Me\\Documents\\my_grid2op_env\\chronics`)

Then you need to fill this `chronics` folder with the data we supposed you had.
You have different ways to achieve this task.

1) The easiest way, in our opinion, is to convert your data into a format that can be understand by
   :class:`grid2op.Chronics.Multifolder` by default (with attribute `gridvalueClass` set to
   :class:`grid2op.Chronics.GridStateFromFile`). So inside your "chronics" folder you should have as many folders
   as their will be different episode on your dataset. And each "episode" folder should contain the files listed
   in the documentation of :class:`grid2op.Chronics.GridStateFromFile`
2) Another way, as always, is to code a class, inheriting from :class:`grid2op.Chronics.GridValue` that is able
   to "load" your file and convert it, when asked, into a valid grid2op format. In this case, the main functions
   to overload are :func:`grid2op.Chronics.GridValue.initialize` (called at the beginning of a scenario)
   and :func:`grid2op.Chronics.GridValue.load_next` call at each "step", each time a new state is generated.


.. _config_py_ex:

Set up the "config.py" file
+++++++++++++++++++++++++++

The goal of this file is to define characteristics for your environment. It is here that you glue everything together.
This file will be loaded each time your environment is created.

This file looks like (example of the "l2rpn_case14_sandbox" one) the one below. Just copy paste it inside your
environment folder `EXAMPLE_FOLDER` (remember, in our example `EXAMPLE_FOLDER` was
`C:\\Users\\Me\\Documents\\my_grid2op_env`). We added some more comment for you to be able to more easily modify it:

.. code-block:: python

    from grid2op.Action import TopologyAndDispatchAction
    from grid2op.Reward import RedispReward
    from grid2op.Rules import DefaultRules
    from grid2op.Chronics import Multifolder
    from grid2op.Chronics import GridStateFromFileWithForecasts
    from grid2op.Backend import PandaPowerBackend

    # you need to define this dictionary.
    config = {
        # type of backend to use, in this example the default PandaPowerBackend
        "backend": PandaPowerBackend,

        # type of action that the agent will be allowed to perform
        "action_class": TopologyAndDispatchAction,

        # use the default Observation class (CompleteObservation)
        "observation_class": None,
        "reward_class": RedispReward,  # which reward function to use

         # how to use the "parameters" of the environment, we don't recommend to change that
        "gamerules_class": DefaultRules,

        # type of chronics, if you used recommended method 1 of the "Organize the "chronics" folder" section
        # don't change that. Otherwise, put the name (and its proper import) of the
        # class you coded
        "chronics_class": Multifolder,

        # this is specific to the "MultiFolder" part. It says that inside each "scenario folder"
        # the data are represented as a format that can be understood by the GridStateFromFileWithForecasts
        # class. You might need to adapt it depending on the choice you made in "Organize the "chronics" folder"
        "grid_value_class": GridStateFromFileWithForecasts,

        # don't change that
        "volagecontroler_class": None,

        # this is used to map the names of the elements from the grid to the chronics data. Typically, the "load
        # connected to substation 1" might have a different name in the grid file (for example in the grid.json)
        # and in the chronics folder (header of the csv if using `GridStateFromFileWithForecasts`)
        "names_chronics_to_grid": None
    }


.. _grid_layout_ex:

Obtain the "grid_layout.json"
++++++++++++++++++++++++++++++

Work in progress.

You can have a look at this file in one of the provided environments for more information.

.. _prod_charac_ex:

Set up the productions and storage characteristics
+++++++++++++++++++++++++++++++++++++++++++++++++++

Work in progress.

Have a look at :func:`grid2op.Backend.Backend.load_redispacthing_data` for productions characteristics and
:func:`grid2op.Backend. Backend.load_storage_data` for storage characteristics.

.. _test_env_ex:

Test your environment
+++++++++++++++++++++

Once the previous steps have been performed, you can try to load your environment in grid2op. This process
is rather easy, but unfortunately, from our own experience, it might not be successful on the first trial.

Anyway, assuming you created your environment in  `EXAMPLE_FOLDER` (remember, in our example `EXAMPLE_FOLDER` was
`C:\\Users\\Me\\Documents\\my_grid2op_env`) you simply need to do, from a python "console" or a python script:

.. code-block:: python

    import grid2op
    env_folder = "C:\\Users\\Me\\Documents\\my_grid2op_env" # or  /home/Me/Documents/my_grid2op_env`
    # in all cases it should match the folder you created and we called EXAMPLE_FOLDER
    # in all this example
    my_custom_env = grid2op.make(env_folder)

    # if it loads, then congrats ! You made your first grid2op environment.

    # you might also need to check things like:
    obs = my_custom_env.reset()

    # and
    obs, reward, done, info = my_custom_env.step(my_custom_env.action_space())


.. note::

    We tried our best to display useful error messages if the environment is not loading properly. If you experience
    any trouble at this stage, feel free to post a github issue on the official grid2op repository
    https://github.com/rte-france/grid2op/issues (you might need to log in on a github account for such purpose)


.. _calibrate_th_lim_ex:

Calibrate the thermal limit
+++++++++++++++++++++++++++

One final (but sometimes important) step for you environment to be really useful is the "calibration of the
thermal limits".

Indeed, the main goal of a grid2op "agent" is to operate the grid "in safety". To that end, you need to specify what
are the "safety criteria". As of writing the main safety criteria are the flows on the powerline (flow in Amps,
"current flow" and not flow in MW).

To complete your environment, you then need to provide for each powerline, the maximum flow allowed on it. This is
optional in the sense that grid2op will work even if you don't do it. But we still strongly recommend to do it.

The way you determine the maximum flow on each powerline is not cover by this "tutorial" as it heavily depends on the
problems you are trying to adress and on the data you have at hands.

Once you have it, you can set it in the "config.py" file. The way you specify it is by setting the
`thermal_limits` key in the `config` dictionary. And this "thermal_limit" is in turn a dictionary, with
the keys being the powerline name, and the value is the associated thermal limit (remember, thermal limit are in A,
not in MW, not in kA).

The example below suppose that you have a powergrid with powerlines named "0_1_0", "0_2_1", "0_3_2", etc.
And that powergrid named "0_1_0" has a thermal limit of `200. A`, that powerline "0_2_1" has a thermal limit
of `300. A`, powerline named "0_3_2" has a thermal limit of `500 A` etc.

.. code-block:: python

    from grid2op.Action import TopologyAction
    from grid2op.Reward import L2RPNReward
    from grid2op.Rules import DefaultRules
    from grid2op.Chronics import Multifolder
    from grid2op.Chronics import GridStateFromFileWithForecasts
    from grid2op.Backend import PandaPowerBackend

    config = {
        "backend": PandaPowerBackend,
        "action_class": TopologyAction,
        "observation_class": None,
        "reward_class": L2RPNReward,
        "gamerules_class": DefaultRules,
        "chronics_class": Multifolder,
        "grid_value_class": GridStateFromFileWithForecasts,
        "volagecontroler_class": None,
        # this part is added compared to the previous example showed in sub section "Set up the "config.py" file"
        # For each powerline (identified by their name, it gives the thermal limit, in A)
        "thermal_limits": {'0_1_0': 200.,
                           '0_2_1': 300.,
                           '0_3_2': 500.,
                           '0_4_3': 600.,
                           '1_2_4': 700.,
                           '2_3_5': 800.,
                           '2_3_6': 900.,
                           '3_4_7': 1000.}
    }

Once done, you should be good to go and doing any study you want with grid2op.
