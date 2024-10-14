.. _model_free_rl:

Model Free Reinforcement Learning
====================================

.. warning::
    This page is in progress. We welcome any contribution :-)

See some example in "l2rpn-baselines" package for now !

The main idea is first to convert the grid2op environment to a gymnasium environment, for example using :ref:`openai-gym`.
And then use some libaries available, 
for example `Stable Baselines <https://stable-baselines3.readthedocs.io/en/master/>`_ or
`RLLIB <https://docs.ray.io/en/latest/rllib/index.html>`_

Some examples are given in "l2rpn-baselines":

- `PPO with RLLIB <https://l2rpn-baselines.readthedocs.io/en/bd-dev/ppo_rllib.html>`_
- `PPO with stable-baselines3 <hhttps://l2rpn-baselines.readthedocs.io/en/bd-dev/ppo_stable_baselines.html>`_

.. include:: final.rst
