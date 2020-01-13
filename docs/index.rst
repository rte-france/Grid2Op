.. Grdi2Op documentation master file, created by
   sphinx-quickstart on Wed Jul 24 15:07:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Grid2Op's technical documentation!
=============================================

Grdi2Op is a pythonic, easy to use framework, to be able to develop, train or evaluate performances of "agent" or
"controler" that acts on a power grid in  different ways.

It is modular and can be use to train reinforcement learning agent or to assess the performance of optimal control
algorithm.

It is flexible and allows the power flow to be computed by the algorithm of your choice. It abstracts the modification
of a powergrid and use this abstraction to compute the **cascading failures** resulting from powerlines disconnection
for example.

**Features**

  - abstract the computation of the cascading failures
  - ability to have the same code running with multiple powerflow calculator
  - parallel execution of one ageent on multiple independent scenarios (multiprocessing)
  - fully customisable: this software has been built to be fully customizable to serve different
    purposes and not only reinforcement learning, or the L2RPN competition.

Main module content
---------------------

.. toctree::
   :maxdepth: 2

   intro
   quickstart
   grid2op
   makeenv
   pltplotly

Technical Documentation
----------------------------

.. toctree::
   :maxdepth: 1

   action
   agent
   backend
   backendpp
   chronicshandler
   environment
   gamerules
   observation
   parameters
   reward
   runner
   space

Main Exceptions
-----------------------
.. toctree::
   :maxdepth: 2

   exception

.. include:: final.rst
