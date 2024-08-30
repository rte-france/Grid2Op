.. |l2rpn_case14_sandbox_layout| image:: ./img/l2rpn_case14_sandbox_layout.png
.. |R2_full_grid| image:: ./img/R2_full_grid.png
.. |l2rpn_neurips_2020_track1_layout| image:: ./img/l2rpn_neurips_2020_track1_layout.png
.. |l2rpn_neurips_2020_track2_layout| image:: ./img/l2rpn_neurips_2020_track2_layout.png
.. |l2rpn_wcci_2022_layout| image:: ./img/l2rpn_wcci_2022_layout.png


Possible workflow to create an environment from existing time series
======================================================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3


Workflow in more details
-------------------------

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

.. include:: final.rst