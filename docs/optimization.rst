.. _optimization_page:

Using "optimization" technique
====================================

.. warning::
    This page is in progress. We welcome any contribution :-)

See some examples in:

- "l2rpn-baselines", for example the `OptimCVXPY <https://l2rpn-baselines.readthedocs.io/en/latest/optimcvxpy.html>`_ agent
- Jan Hendrick Menke `PP_Baseline` an optimizer based on pandapower OPF available here  `PandapowerOPFAgent <https://github.com/jhmenke/grid2op_pp_baseline>`_
- An optimizer developed by RTE able to change the topology `MILP Agent <https://github.com/rte-france/grid2op-milp-agent>`_

Basically an "optimizer" agent looks like (from a very high level):

1) have a simplification of the "MDP" / decision process in the shape of an optimziation problem
2) make a formulation of this problem using a "framework" preferably in python (*eg* using `pandapower`, `cvxpy` or `or-tools`)
3) update the "formulation" using the observation received
4) run a solver to solve the "problem" 
5) convert back the "decisions" (output) of the solver into a "grid2op" action

.. include:: final.rst
