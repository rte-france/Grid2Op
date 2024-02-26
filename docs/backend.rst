.. currentmodule:: grid2op.Backend

.. _backend-module:

Backend
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------

.. warning:: Backends are internal to grid2op. You should not have to recode any backend if you are "simply"
    using grid2op, for example to develop new controller.

    Backend is an abstraction that represents the physical system (the powergrid). In theory every powerflow can be
    used as a backend. For now we only provide a Backend that uses  `Pandapower <http://www.pandapower.org/>`_ and
    a port in c++ to a subset of pandapower called `LightSim2Grid <https://github.com/BDonnot/lightsim2grid>`_ .

    Both can serve as example if you want to code a new backend.

This Module defines the template of a backend class.

Backend instances are responsible to translate action into
comprehensive powergrid modifications that can be process by your "Simulator".
The simulator is responsible to perform the powerflow (AC or DC or Time Domain / Dynamic / Transient simulation)
and to "translate back" the results (of the simulation) to grid2op.

More precisely, a backend should:

#. inform grid2op of the grid: which objects exist, where are they connected etc.
#. being able to process an object of type :class:`grid2op.Action._backendAction._BackendAction`
   into some modification to your solver (*NB* these "BackendAction" are created by the :class:`grid2op.Environment.BaseEnv`
   from the agent's actions, the time series modifications, the maintenances, the opponent, etc. The backend **is not**
   responsible for their creation)
#. being able to run a simulation (DC powerflow, AC powerflow or time domain / transient / dynamic)
#. expose (through some functions like :func:`Backend.generators_info` or :func:`Backend.loads_info`) 
   the state of some of the elements in the grid.

.. note::
  A backend can model more elements than what can be controlled or modified in grid2op.
  For example, at time of writing, grid2op does not allow the modification of 
  HVDC powerlines. But this does not mean that grid2op will not work if your grid
  counts such devices. It just means that grid2op will not be responsible
  for modifying them. 

.. note::
  A backend can expose only part of the grid to the environment / agent. For example, if you
  give it as input a pan european grid but only want to study the grid of Netherlands or
  France your backend can only "inform" grid2op (in the :func:`Backend.load_grid` function)
  that "only the Dutch (or French) grid" exists and leave out all other informations.

  In this case grid2op will perfectly work, agents and environment will work as expected and be 
  able to control the Dutch (or French) part of the grid and your backend implementation
  can control the rest (by directly updating the state of the solver).

It is also through the backend that some quantities about the powergrid (such as the flows) can be inspected.

A backend is mandatory for a Grid2Op environment to work properly.

To be a valid backend, some properties are mandatory:

    - order of objects matters and should be deterministic (for example :func:`Backend.get_line_status`
      shall return the status of the lines always in the same order)
    - order of objects should be the same if the same underlying object is queried (for example, is
      :func:`Backend.get_line_status`\[i\] is the status of the powerline "*toto*", then
      :func:`Backend.get_thermal_limit`\[i\] returns the thermal limits of this same powerline "*toto*")
    - it allows to compute AC and DC powerflow
    - it allows to:

        - change the value consumed (both active and reactive) by each load of the network
        - change the amount of power produced and the voltage setpoint of each generator unit of the powergrid
        - allow for powerline connection / disconnection
        - allow for the modification of the connectivity of the powergrid (change in topology)
        - allow for deep copy.

The order of the values returned are always the same and determined when the backend is loaded by its attribute
'\*_names'. For example, when the ith element of the results of a call to :func:`Backend.get_line_flow` is the
flow on the powerline with name `lines_names[i]`.

Creating a new backend
-----------------------
We developed a dedicated page for the development of new "Backend" compatible with grid2op here
:ref:`create-backend-module`.

Detailed Documentation by class
-------------------------------

Then the `Backend` module:

.. automodule:: grid2op.Backend
    :members:
    :private-members:
    :special-members:
    :autosummary:


.. include:: final.rst
