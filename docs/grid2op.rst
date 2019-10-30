Grid2Op module
===================================
.. module:: grid2op

The grid2op module allows to perform sequential action on a powergrid.

It is modular in the sens that it allows to use different powerflow solver. It proposes an internal representation
of the data that can be feed to powergrids and multiple class to specify how it's done.

For example, it is possible to use an "action" to set the production value of some powerplant. But we
also know that it's not possible to do this for every powerplant (for example, asking a windfarm to produce more
energy is not possible: the only way would be to increase the speed of the wind). It is possible to implement
these kind of restrictions in this "game like" environment.

Today, the main usage of this plateform is to serve as a computation engine for the `L2RPN <www.l2rpn.chalearn.com>`_
cpompetitions.

This plateform is still under development. If you notice a bug, let us know with a github issue at
`Grid2Op <https://github.com/rte-france/Grid2Op>`_

####################
Glossary
####################
TODO Coming soon