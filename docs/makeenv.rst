.. currentmodule:: grid2op.MakeEnv
.. _make-env-module:

Make: Using pre defined Environments
====================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
The function define in this module is the easiest and most convenient ways to create a valid
:class:`grid2op.Environment.Environment`.

To get started with such an environment, you can simply do:

.. code-block:: python

    import grid2op
    env = grid2op.make("l2rpn_case14_sandbox")


You can consult the different notebooks in the `getting_stared` directory of this package for more information on
how to use it.

Created Environment should behave exactly like a gymnasium environment. If you notice any unwanted behavior, please address
an issue in the official grid2op repository: `Grid2Op <https://github.com/Grid2Op/grid2op>`_

The environment created with this method should be fully compatible with the gymnasium framework: if you are developing
a new algorithm of "Reinforcement Learning" and you used the openai gymnasium framework to do so, you can port your code
in a few minutes (basically this consists in adapting the input and output dimension of your BaseAgent) and make it work
with a Grid2Op environment. An example of such modifications is exposed in the getting_started/ notebooks.

.. _usage:

Usage
------

There are two main ways to use the :func:`make` function. The first one is to directly pass the name of the environment
you want to use:

.. code-block:: python

    import grid2op
    env = grid2op.make("l2rpn_case14_sandbox")

This will create the environment known "l2rpn_case14_sandbox" with all default parameters. If this environment is has
not been downloaded, at the first call to this function it will download it and store it in a cache on your system (
see the section :ref:`cache_manip` for more information), afterwards it will use the downloaded environment.

If your computer don't have internet access, or you prefer to download things manually, it is also possible to provide
the full absolute path of you dataset. On linux / unix (including macos) machines this will be something like

.. code-block:: python

    import grid2op
    env = grid2op.make("/full/path/where/the/env/is/located/l2rpn_case14_sandbox")

And on windows based machine this will look like:

.. code-block:: python

    import grid2op
    env = grid2op.make("C:\\where\\the\\env\\is\\located\\l2rpn_case14_sandbox")

In bot cases it will load the environment named "l2rpn_case14_sandbox" (provided that you found a way to get it on your
machine) located at the path "/full/path/where/the/env/is/located/l2rpn_case14_sandbox" (or
"C:\\the\\full\\path\\where\\the\\env\\is\\located\\l2rpn_case14_sandbox").

Important notes
---------------
As of version 0.8.0 a ":func:`make`" has been updated in grid2op. This function, replace the current implementation of
renamed :func:`make_old`, merges the behaviour of "grid2op.download" script and "make_old" function.

It has the following behavior:

1) if you specify a full path to a local environment (containing the chronics and the default parameters),
   it will be used (see section :ref:`usage`)
2) if you specify the name of an environment that you have already downloaded, it will use this environment (NB
   currently no checks are implemented if the environment has been updated remotely, which can happen if
   we realize there were some issues with it.). If you want to update the environments you downloaded
   please use :func:`grid2op.update_env()`
3) you are expected to provide an environment name (if you don't know what this is just put `"l2rpn_case14_sandbox"`)
4) if the flag `test` is set to ``False`` (default behaviour) and none of the above conditions are met, the
   :func:`make` will download the data of this environment locally the first time it is called. If you don't want
   to download anything then you can pass the flag ``test=True`` (in this case only a small sample of 
   time series will be available. We don't recommend to do that at all !)
5) if ``test=True`` (NON default behaviour) nothing will be loaded, and the :func:`make` will attempt to use a
   pre defined environment provided with the python package. We want to emphasize that because the environments provided
   with this package contains only little data, they are not suitable for leaning a consistent agent / controler. That
   is why a warning is sent in this case. Also, keep in mind that if you don't pass ``test=True`` then you will not
   have the possibility to search for these environments provided in the package. *Setting "test=True" is NOT
   recommended* for most usage. Have a look at the section :ref:`usage` for more details on how to use `make`,
   especially if you don't have an internet connection.
6) if no valid environment is found, :func:`make` throws a EnvError.

.. _cache_manip:

Cache manipulation
-------------------------------
Editing the file ``~/.grid2opconfig.json`` allows you to change the data cache location. Programatically, it can be
done with :func:`change_local_dir`.

Call :func:`get_current_local_dir` to get the local cache directory location.

You can list the environments in the local cache directory by calling :func:`list_available_local_env` and list all
environments that can be downloaded with :func:`list_available_remote_env` (*nb* `list_available_remote_env` requires
an internet connection)

.. code-block:: python

    import grid2op
    print("The current local directory where the environment are downloaded is \n{}"
          "".format(grid2op.get_current_local_dir()))
    print("The environments available without necessary download are: \n{}"
          "".format(grid2op.list_available_local_env()))
    print("I can download these environments from the internet: \n{}"
          "".format(grid2op.list_available_remote_env()))

**NB** if you change the cache directory, all previously downloaded environments will not be visible by grid2op and they
will not be removed from your local hard drive. This is why we don't recommend to change this folder unless you have a
valid reason to do so.

.. _env_cust_makeenv:

Customize your environment
--------------------------
When you create it, you can change different parameters of the environments. We summarize all parameters
that can be modified at the creation of your environment. We recommend you to see the section
`Parameters` of the :func:`make_from_dataset_path`
for more information about the effect of this attributes. **NB** arguments preceding by a \* are listed to be
exhaustive. They are technical arguments and should not be modified unless you have a reason to. For example, in the
context of the L2RPN competition, we don't recommend to modify them.

- `dataset_path`: used to specify the name (or the path) of the environment you want to load
- `backend`: a initialized backend that will carry out the computation related to power system [mainly use if you want
  to change from PandapowerBackend (default) to a different one *eg* LightSim2Grid]
- `reward_class`: change the type of reward you want to use for your agent (see section 
  :ref:`reward-module` for more information).
- `other_reward`: tell "env.step" to return addition "rewards"(see section 
  :ref:`reward-module` for more information).
- `difficulty`, `param`: control the difficulty level of the game (might not always be available)
- `chronics_class`, `data_feeding_kwargs`: further customization to how the data will be generated,
  see section :ref:`environment-module-data-pipeline` for more information
- `n_busbar`: (``int``, default 2) [new in version 1.9.9]  see section :ref:`substation-mod-el` 
  for more information
- \* `chronics_path`, `data_feeding`, : to overload default path for the data (**not recommended**)
- \* `action_class`: which action class your agent is allowed to use (**not recommended**).
- \* `gamerules_class`: the rules that are checked to declare an action legal / illegal (**not recommended**)
- \* `volagecontroler_class`: how the voltages are set on the grid (**not recommended**)
- \* `grid_path`: the path where the default powergrid properties are stored (**not recommended**)
- \* `observation_class`, `kwargs_observation`: which type of observation do you use (**not recommended**)
- \* `opponent_action_class`, `opponent_class`, `opponent_init_budget`, `opponent_budget_per_ts`,
  `opponent_budget_class`, `opponent_space_type`, `kwargs_opponent`: all configuration for the opponent. (**not recommended**)
- \* `has_attention_budget`, `attention_budget_class`, `kwargs_attention_budget`: all configuration
   for the "alarm" / "attention budget" parameters. (**not recommended**)

More information about the "customization" of the environment, especially to optimize the I/O or to manipulate
which data you interact with are available in the :ref:`environment-module` module (:ref:`environment-module-usage` section).


.. warning:: Don't modify the action class

    We do not recommend to modify the keyword arguments starting with \*, and especially the action_class.

You can customize an environment with:

.. code-block:: python

    import grid2op
    env = grid2op.make(dataset_path, 
                       backend=...,  # put a compatible backend here
                       reward_class=...,  # change the reward function, see BaseReward
                       other_reward={key: reward_func}, # with `key` being strings and `reward_func` inheriting from BaseReward
                       difficulty=...,  # str or ints
                       param=...,  # any Parameters (from grid2op.Parameters import Parameters)
                       etc.
                       )

See documentation of :func:`grid2op.MakeEnv.make_from_dataset_path` for more information about all these parameters.

Detailed Documentation by class
--------------------------------

.. automodule:: grid2op.MakeEnv
    :members:
    :autosummary:

.. include:: final.rst
