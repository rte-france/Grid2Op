.. currentmodule:: grid2op.Chronics.handlers

.. _tshandler-module:

Time Series Handlers
======================

.. versionadded:: 1.9.0
    This module is new in version 1.9.0

The goal of this module is to allow some fine-grain manipulation over the generation
of the "time series" data of the environment.

You might want to first read :ref:`doc_timeseries` to get familiar with what are these 
time series and how they are used.

With these "handlers" you can generate the data used by the environment independantly
for each type. To use this you have to provide :

- one handler for "load_p"
- one handler for "load_q"
- one handler for "prod_p"
- one handler for "prod_v"
- (optional) one handler for the "maintenance"
- (optional) one handler for "load_p_forecasted"
- (optional) one handler for "load_q_forecasted"
- (optional) one handler for "prod_p_forecasted"
- (optional) one handler for "prod_v_forecasted"

.. warning:: 
    I will not write "*with great power comes great ...*" (close enough, you got it)
    but not every handlers are compatible with every others and depending 
    on the options that you set (*e.g* "chunk_size") you might end up with
    different results.

    If you use handlers you need at least to understand the basics of 
    what you are doing.

Interests of "handlers"
--------------------------

The main interest (at time of writing) of handlers is to be able to use
some "approximation" of "multi steps ahead forecasts" in already available environment.

You can do that by using the :class:`PerfectForecastHandler` or 
:class:`NoisyForecastHandler` for "load_p_forecasted", "load_q_forecasted", 
"prod_p_forecasted" or "prod_v_forecasted".

For example, for the environment `l2rpn_wcci_2022` this gives:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler, DoNothingHandler
    env_name = "l2rpn_wcci_2022"
    forecasts_horizons = [5, 10, 15, 20, 25, 30]

    env = grid2op.make(env_name,
                       data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": DoNothingHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "h_forecast": forecasts_horizons,
                                            "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": PerfectForecastHandler("load_q_forecasted"),
                                           }
                      )

    obs = env.reset()

    # all works
    obs.simulate(env.action_space(), 1)
    obs.simulate(env.action_space(), 2)
    obs.simulate(env.action_space(), 6)

    sim_obs1, *_ = obs.simulate(env.action_space())
    sim_obs2, *_ = sim_obs1.simulate(env.action_space())
    sim_obs3, *_ = sim_obs2.simulate(env.action_space())

    # and even this
    f_env = obs.get_forecast_env()
    obs = f_env.reset()
    obs.max_step == 7  # initial state + 6 steps ahead forecast

This was not possible with the intial environment (and its default `data_feeding_kwargs`)

.. note::
    Above, the forecast are "perfect" meaning the agent sees what will be the exact 
    load and geenration a few hours in advance. If you want to add some noise, 
    you can use the :class:`NoisyForecastHandler`

Standard data generation and equivalent handlers
--------------------------------------------------

For each type of data generation, you can use "handler" to retrieve the
same results. It has little interest to make the handlers if you want to 
get the default behaviour of course (more verbose and slower) but it might allow
to use different data (for example removing the maintenance, or increasing the 
forecast horizons etc.)

- :class:`grid2op.Chronics.ChangeNothing`: 
  
  - "load_p_handler" : :class:`DoNothingHandler`
  - "load_q_handler" : :class:`DoNothingHandler`
  - "gen_p_handler" : :class:`DoNothingHandler`
  - "gen_v_handler" : :class:`DoNothingHandler`

- :class:`grid2op.Chronics.GridStateFromFile`: 
  
  - "load_p_handler" : :class:`CSVHandler`
  - "load_q_handler" : :class:`CSVHandler`
  - "gen_p_handler" : :class:`CSVHandler`
  - "gen_v_handler" : :class:`CSVHandler`
  - "maintenance_handler" : :class:`CSVMaintenanceHandler`

- :class:`grid2op.Chronics.GridStateFromFileWithForecasts`: 
  
  - "load_p_handler" : :class:`CSVHandler`
  - "load_q_handler" : :class:`CSVHandler`
  - "gen_p_handler" : :class:`CSVHandler`
  - "gen_v_handler" : :class:`CSVHandler`
  - "maintenance_handler" : :class:`CSVMaintenanceHandler`
  - "load_p_for_handler" : :class:`CSVForecastHandler`
  - "load_q_for_handler" : :class:`CSVForecastHandler`
  - "gen_p_for_handler" : :class:`CSVForecastHandler`
  - "gen_v_for_handler" : :class:`CSVForecastHandler`

- :class:`grid2op.Chronics.GridStateFromFileWithForecastsWithMaintenance`: 
  
  - "load_p_handler" : :class:`CSVHandler`
  - "load_q_handler" : :class:`CSVHandler`
  - "gen_p_handler" : :class:`CSVHandler`
  - "gen_v_handler" : :class:`CSVHandler`
  - "maintenance_handler" : :class:`JSONMaintenanceHandler`
  - "load_p_for_handler" : :class:`CSVForecastHandler`
  - "load_q_for_handler" : :class:`CSVForecastHandler`
  - "gen_p_for_handler" : :class:`CSVForecastHandler`
  - "gen_v_for_handler" : :class:`CSVForecastHandler`

- :class:`grid2op.Chronics.GridStateFromFileWithForecastsWithoutMaintenance`: 
  
  - "load_p_handler" : :class:`CSVHandler`
  - "load_q_handler" : :class:`CSVHandler`
  - "gen_p_handler" : :class:`CSVHandler`
  - "gen_v_handler" : :class:`CSVHandler`
  - "maintenance_handler" : :class:`DoNothingHandler` (or more simply `None`)
  - "load_p_for_handler" : :class:`CSVForecastHandler`
  - "load_q_for_handler" : :class:`CSVForecastHandler`
  - "gen_p_for_handler" : :class:`CSVForecastHandler`
  - "gen_v_for_handler" : :class:`CSVForecastHandler`


For example instead of having (imports are removed for clarity):

.. code-block:: python
    
    env = grid2op.make(env_name, 
                       data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecasts,}
                      )

You can write, to have similar results:

.. code-block:: python
    
    env = grid2op.make(env_name, 
                       data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": CSVHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "maintenance_handler": CSVMaintenanceHandler(),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                           }
    )

.. warning::
    The behaviour is not exactly the same between the "GridStateFromFileXXX" and the "FromHandlers" especially in the case of
    stochastic maintenance (:class:`grid2op.Chronics.GridStateFromFileWithForecastsWithMaintenance`). Indeed when you 
    call `env.seed(XXX)` the maintenance generated by `GridStateFromFileWithForecastsWithMaintenance` will not
    be the same as the one generated by `JSONMaintenanceHandler` (because the underlying pseudo random number generator)
    will not be seeded the same way.


Use with standard environments
--------------------------------

In this section we write the "handlers" version you can use to reproduce the behaviour of 
some grid2op environments.

l2rpn_case14_sandbox
+++++++++++++++++++++++

The setting for the "l2rpn_case14_sandbox" is:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import CSVHandler, CSVForecastHandler

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name, 
                        data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": CSVHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                            }
                        )
    obs = env.reset()
    # continue like you would normally ...

You can now tweak it to add more forecast or add / remove some maintenance by modifying the `XXX_for_handler` or by adding the `maintenance_handler` for example.

l2rpn_wcci_2022
+++++++++++++++++

The setting for the "l2rpn_wcci_2022" is:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import CSVHandler, JSONMaintenanceHandler, CSVForecastHandler, DoNothingHandler

    env_name = "l2rpn_wcci_2022"
    env = grid2op.make(env_name, 
                        data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": DoNothingHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "maintenance_handler": JSONMaintenanceHandler(),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                            }
                        )
    obs = env.reset()
    # continue like you would normally ...

You can now tweak it to add more forecast or add / remove some maintenance by modifying the `XXX_for_handler` or by removing the `maintenance_handler` for example.

l2rpn_icaps_2021
++++++++++++++++++

The setting for the "l2rpn_icaps_2021" is:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import CSVHandler, JSONMaintenanceHandler, CSVForecastHandler

    env_name = "l2rpn_icaps_2021_small"  # or "l2rpn_icaps_2021_large"
    env = grid2op.make(env_name, 
                        data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": CSVHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "maintenance_handler": JSONMaintenanceHandler(),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                            }
                        )
    obs = env.reset()
    # continue like you would normally ...

You can now tweak it to add more forecast or add / remove some maintenance by modifying the `XXX_for_handler` or by removing the `maintenance_handler` for example.


l2rpn_neurips_2020_track1
++++++++++++++++++++++++++

The setting for the "l2rpn_neurips_2020_track1" is:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import CSVHandler, JSONMaintenanceHandler, CSVForecastHandler

    env_name = "l2rpn_neurips_2020_track1_small"  # or "l2rpn_neurips_2020_track1_large"
    env = grid2op.make(env_name, 
                        data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": CSVHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "maintenance_handler": JSONMaintenanceHandler(),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                            }
                        )
    obs = env.reset()
    # continue like you would normally ...

You can now tweak it to add more forecast or add / remove some maintenance by modifying the `XXX_for_handler` or by removing the `maintenance_handler` for example.


l2rpn_neurips_2020_track1
++++++++++++++++++++++++++

The setting for the "l2rpn_neurips_2020_track2" is:

.. code-block:: python

    import grid2op
    from grid2op.Chronics import FromHandlers
    from grid2op.Chronics.handlers import CSVHandler, JSONMaintenanceHandler, CSVForecastHandler

    env_name = "l2rpn_neurips_2020_track2_small"  # or "l2rpn_neurips_2020_track2_large"
    env = grid2op.make(env_name, 
                        data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                            "gen_p_handler": CSVHandler("prod_p"),
                                            "load_p_handler": CSVHandler("load_p"),
                                            "gen_v_handler": CSVHandler("prod_v"),
                                            "load_q_handler": CSVHandler("load_q"),
                                            "maintenance_handler": JSONMaintenanceHandler(),
                                            "gen_p_for_handler": CSVForecastHandler("prod_p_forecasted"),
                                            "load_p_for_handler": CSVForecastHandler("load_p_forecasted"),
                                            "load_q_for_handler": CSVForecastHandler("load_q_forecasted"),
                                            }
                        )
    obs = env.reset()
    # continue like you would normally ...

You can now tweak it to add more forecast or add / remove some maintenance by modifying the `XXX_for_handler` or by removing the `maintenance_handler` for example.


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Chronics.handlers
    :members:
    :autosummary:

.. include:: final.rst
