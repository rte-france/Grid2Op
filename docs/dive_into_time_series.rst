.. _chornix2grid_github: https://github.com/bdonnot/chronix2grid

.. _doc_timeseries:

Input data of an environment
===================================

A grid2op "environment" is nothing more than a local folder on your computer. 

This folder consists of different things:

- a file representing a powergrid (for available environment at time of writing `grid.json`) and
  a "model" to make sense of this file (usable in grid2op thanks to a :ref:`backend-module`)
- **FOCUS OF THIS FILE** some "input time series" for the generation and the load
- some other optional files, such as:

  - the characteristics of loads
  - of generators
  - of storage units 
  - or the "geographical" coordinates of the elements of the grid for example.

Role of these time series
---------------------------------

Grid2op model the real time behaviour of a powergrid. It supposes that somehow the amount of
power consumed by each consumers is known as well as the amount of power each generator will produce.

In the case of grid2op, we suppose that all of these are "known" by the environment but unknown to 
the agent. More specifically, the "agent" / "actor" / "controler" knows 
(thanks to the :class:`grid2op.Observation.BaseObservation`) the amount of power produce at 
a given time but not exactly what will be produced / consumed in the future. This is one of
the reason why "l2rpn environments" are "partially available" environments.

Currently, the package used to generate these time series is called `chornix2grid_github`_
but anything that generate data for all loads / generators could be used as "input" for a grid2op environment.

.. note::

  There are differences between the "theorical time series" generated that serve as input to a 
  grid2op environment and the one the agent will see. Among the differences:

  - redispatching (a type of action performed by an agent) will modify the production of controlable generation 
    units such as thermal, nuclear or hydro
  - curtailment (another type of action performed by an agent) can decrease the production of generator using 
    renewable energy sources
  - storage unit (another type of action) will have some impact on the generators. A fondamental "law" of energy 
    grid is that it should be balanced at all time. If a storage units inject on the 
    grid xxx MW then "something else" should decrease its production of xxx MW. In grid2op we suppose that this
    "something else" is controlable generators.
  - modification of topology or any other type of actions (and sometimes even "non action") will impact the
    electrical losses on the grid. And because the "fondamental law of energy grid" (power injected should match exactly
    power removed and power lost) a set of special generators (they have a fancy name: "generators participating 
    to the slack bus") are modified by the :class:`grid2op.Backend.Backend`.

.. note::

  Depending on the environment, some "forecast" are available. Forecasts represent a view at a given 
  time for the near future. In most environment the forecast are available for the next step.

  You can however, thanks to :ref:`tshandler-module` "generate" forecasts for longer horizons if
  you need them.


Different type of "data"
-----------------------------

To run a grid2op environment, you need to provide different type of data to the :class:`grid2op.Environment.Environment`,
the data includes:

- "load_p": active consumption for all loads at all steps of the environment (in MW)
- "load_q": reactive consumption for all loads and at all steps of the environment (in MVAr)
- "gen_p" / "prod_p": active production for all generators at all steps of the environment (in MW)
- "gen_v" / "prod_v": voltage setpoint for each geneators at all steps of the environment (in kV)
- (optional) "load_p_forecasted": some forecast of "a few" steps ahead for "load_p" (in MW)
- (optional) "load_q_forecasted": some forecast of "a few" steps ahead for "load_q" (in MVAr)
- (optional) "prod_p_forecasted": some forecast of "a few" steps ahead for "prod_p" (in MW)
- (optional) "prod_v_forecasted": some forecast of "a few" steps ahead for "prod_v" (in kV)
- (optional) "maintenance": whether some powerlines will be out of service for maintenance operations

.. note::
    \*\*\*\_forecasted will be the data used by the :func:`grid2op.Observation.BaseObservation.simulate`,
    :func:`grid2op.Observation.BaseObservation.get_forecasted_inj` 
    or the or the :func:`grid2op.Observation.BaseObservation.get_forecast_env` functions.

.. note::
  On some environment, if you dont' want to use :func:`grid2op.Observation.BaseObservation.simulate`
  or :func:`grid2op.Observation.BaseObservation.get_forecast_env` and want to speed up 
  the time of the "step" function, you can call :func:`grid2op.Environment.BaseEnv.deactivate_forecast` function.

Available classes
-------------------------

The main classes you can use are:

1) read the environment data ("load_p", "load_q", "prod_p", "prod_v") from csv (
   for example generated with `chornix2grid_github`_), 
   see :class:`grid2op.Chronics.GridStateFromFile`
2) If you want to read this data but also to read some
   forecasts ("load_p_forecasted", "load_q_forecasted", "prod_p_forecasted", "prod_v_forecasted") 
   for next steps, you can use :class:`grid2op.Chronics.GridStateFromFileWithForecasts`
3) If you want also to generate maintenance data "on the fly" (that varies between episode then)
   you might want to see :class:`grid2op.Chronics.GridStateFromFileWithForecastsWithMaintenance`
4) But if you want to deactivate the maintenance even if some files describing the maintenance are 
   present in your "chronics" folder, have a look at:
   :class:`grid2op.Chronics.GridStateFromFileWithForecastsWithoutMaintenance`
5) Previous classes will rely on data being generated and available on the hard drive. This means
   than the same data will be used over and over again, you might want to consider Generating
   new data on the fly
   :class:`grid2op.Chronics.FromChronix2grid`  [**NB** we DO NOT recommend to do that, it might takes 
   a few minutes at each call to :func:`grid2op.Environment.Environment.reset` and dramatically slow
   down your training]
6) And if you want some more control on what is being done, you can have a look at the 
   dedicated module: :ref:`tshandler-module`

Example of such times series "generator"
----------------------------------------

TODO


.. include:: final.rst
