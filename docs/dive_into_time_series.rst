
Input data of an environment
===================================

A grid2op "environment" consists in different things:
- a file representing a powergrid (for available environment at time of writing `grid.json`) and
  a "model" to make sense of this file (usable in grid2op thanks to a :ref:`backend-module`)
- some coordinate
- some storage units and generators description
- **FOCUS OF THIS FILE** some "input time series" for the generation and the load

Role of these time series
---------------------------------

Grid2op model the real time behaviour of a powergrid. It supposes that 

Similar to "sprite" in a more standard video game, to "level layout" or things like that.

Available classes
-------------------------

TODO:

1) read the things from csv (csv generated with the method you want, for example chronix2grid)
2) random things such as maintenance (can vary between episode, even if the rest is fixed)
3) Forecast
4) from "online generation" with the appropriate class
5) from "handler"

Example of such times series generator
----------------------------------------

TODO


.. include:: final.rst
