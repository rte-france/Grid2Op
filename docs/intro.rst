Introduction
===================================
Grid2Op is a tool that allows to perform Reinforcement Learning (abbreviated RL) or any
other time dependant simulation of steady state powerflow.

The simulator is able to emulate a power grid (of any size or characteristics) subject to a set of
temporal injections (productions and consumptions) or maintenance / hazards for discretized
timesteps.

Loadflow computation are carried out using any Backend you wish. A default backend, relying
on the open source `pandapower <https://pandapower.readthedocs.io/en/latest/about.html>`_
library is available as an example.

Any other tools that is able to perform power flow computation can be used as a "backend" to
play the game or to accelerate the training. Instructions and method to implement
a new backend are available in the :class:`Grid2Op.Backend` documentation.

TO be continued (package still under development)
