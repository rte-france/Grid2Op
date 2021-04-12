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
Backend instances are responsible to translate action (performed either by an BaseAgent or by the Environment) into
comprehensive powergrid modifications.
They are responsible to perform the powerflow (AC or DC) computation.

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
.. automodule:: grid2op.Backend
    :members:
    :private-members:
    :special-members:
    :autosummary:


.. include:: final.rst
