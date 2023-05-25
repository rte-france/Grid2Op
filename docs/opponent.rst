.. currentmodule:: grid2op.Opponent

Opponent Modeling
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
Power systems are a really important tool today, that can be as resilient as possible to avoid possibly dramatic
consequences.

In grid2op, we chose to enforce this property by implementing an "Opponent" (modeled thanks to the :class:`BaseOpponent`
that can take malicious actions to deteriorate the state of the powergrid and make tha Agent (:class:`grid2op.Agent`)
fail. To make the agent "game over" is really easy (for
example it could isolate a load by forcing the disconnection of all the powerline that powers it). This would not be
fair, and that is why the Opponent has some dedicated budget (modeled with the :class:`BaseActionBudget`).

The class :class:`OpponentSpace` has the delicate role to:
- send the necessary information for the Opponent to attack properly.
- make sure the attack performed by the opponent is legal
- compute the cost of such attack
- make sure this cost is not too high for the opponent budget.

How to create an opponent in any environment
---------------------------------------------

This section is a work in progress, it will only cover how to set up one type of opponent, and supposes
that you already know which lines you want to attack, at which frequency etc.

More detailed information about the opponent will be provide in the future.

The set up for the opponent in the "l2rpn_neurips_track1" has the following configuration.

.. code-block:: python

    lines_attacked = ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131", "39_41_121",
                  "43_44_125", "44_45_126", "34_35_110", "54_58_154"]
    rho_normalization = [0.45, 0.45, 0.6, 0.35, 0.3, 0.2,
                         0.55, 0.3, 0.45, 0.55]
    opponent_attack_cooldown = 12*24  # 24 hours, 1 hour being 12 time steps
    opponent_attack_duration = 12*4  # 4 hours
    opponent_budget_per_ts = 0.16667  # opponent_attack_duration / opponent_attack_cooldown + epsilon
    opponent_init_budget = 144.  # no need to attack straightfully, it can attack starting at midday the first day
    config = {
        "opponent_attack_cooldown": opponent_attack_cooldown,
        "opponent_attack_duration": opponent_attack_duration,
        "opponent_budget_per_ts": opponent_budget_per_ts,
        "opponent_init_budget": opponent_init_budget,
        "opponent_action_class": PowerlineSetAction,
        "opponent_class": WeightedRandomOpponent,
        "opponent_budget_class": BaseActionBudget,
        'kwargs_opponent': {"lines_attacked": lines_attacked,
                            "rho_normalization": rho_normalization,
                            "attack_period": opponent_attack_cooldown}
    }

To create the same type of opponent on the **case14** grid you can do:

.. code-block:: python

    import grid2op
    from grid2op.Action import PowerlineSetAction
    from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
    env_name = "l2rpn_case14_sandbox"

    env_with_opponent = grid2op.make(env_name,
                                     opponent_attack_cooldown=12*24,
                                     opponent_attack_duration=12*4,
                                     opponent_budget_per_ts=0.5,
                                     opponent_init_budget=0.,
                                     opponent_action_class=PowerlineSetAction,
                                     opponent_class=RandomLineOpponent,
                                     opponent_budget_class=BaseActionBudget,
                                     kwargs_opponent={"lines_attacked":
                                          ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]}
                                     )
    # and now you have an opponent on the l2rpn_case14_sandbox
    # you can for example
    obs = env_with_opponent.reset()

    act = ...  # chose an action here
    obs, reward, done, info = env_with_opponent.step(act)


And for the track2 of neurips, if you want to make it even more complicated, you can add an opponent
in the same fashion:

.. code-block:: python

    import grid2op
    from grid2op.Action import PowerlineSetAction
    from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
    env_name = "l2rpn_neurips_2020_track2_small"

    env_with_opponent = grid2op.make(env_name,
                                     opponent_attack_cooldown=12*24,
                                     opponent_attack_duration=12*4,
                                     opponent_budget_per_ts=0.5,
                                     opponent_init_budget=0.,
                                     opponent_action_class=PowerlineSetAction,
                                     opponent_class=RandomLineOpponent,
                                     opponent_budget_class=BaseActionBudget,
                                     kwargs_opponent={"lines_attacked":
                                                         ["26_31_106",
                                                          "21_22_93",
                                                          "17_18_88",
                                                          "4_10_162",
                                                          "12_14_68",
                                                          "14_32_108",
                                                          "62_58_180",
                                                          "62_63_160",
                                                          "48_50_136",
                                                          "48_53_141",
                                                          "41_48_131",
                                                          "39_41_121",
                                                          "43_44_125",
                                                          "44_45_126",
                                                          "34_35_110",
                                                          "54_58_154",
                                                          "74_117_81",
                                                          "80_79_175",
                                                          "93_95_43",
                                                          "88_91_33",
                                                          "91_92_37",
                                                          "99_105_62",
                                                          "102_104_61"]}
                                     )
    # and now you have an opponent on the l2rpn_case14_sandbox
    # you can for example
    obs = env_with_opponent.reset()

    act = ...  # chose an action here
    obs, reward, done, info = env_with_opponent.step(act)

To summarize what is going on here:

- `opponent_attack_cooldown`: give the minimum number of time between two attacks (here 1 attack per day)
- `opponent_attack_duration`: duration for each attack (when a line is attacked, it will not be possible to reconnect
  it for that many steps). In the example it's 4h (so 48 steps)
- `opponent_action_class`: type of the action the opponent will perform (in this case `PowerlineSetAction`)
- `opponent_class`: type of the opponent. Change it at your own risk.
- `opponent_budget_class`: Each attack will cost some budget to the opponent. If no budget, the opponent cannot
  attack. This specifies how the budget are computed. Do not change it.
- `opponent_budget_per_ts`: increase of the budget of the opponent per step. The higher this number, the faster the
  the opponent will regenerate its budget.
- `opponent_init_budget`: initial opponent budget. It is set to 0 to "give" the agent a bit of time before the opponent
  is triggered.
- `kwargs_opponent`: additional information for the opponent. In this case we provide for each grid the powerline it
  can attack.

.. note::

    This is only valid for the `RandomLineOpponent` that disconnect powerlines randomly (but not uniformly!). For other
    type of Opponent, we don't provide any information in the documentation at this stage. Feel free to submit
    a github issue if this is an issue for you.

How to deactivate an opponent in an environment
--------------------------------------------------

If you come accross an environment with an "opponent" already present but for some reasons you want to
deactivate it, you can do this by customization the call to "grid2op.make" like this:

.. code-block:: python

  import grid2op
  from grid2op.Action import DontAct
  from grid2op.Opponent import BaseOpponent, NeverAttackBudget
  env_name = ...

  env_without_opponent = grid2op.make(env_name,
                                      opponent_attack_cooldown=999999,
                                      opponent_attack_duration=0,
                                      opponent_budget_per_ts=0,
                                      opponent_init_budget=0,
                                      opponent_action_class=DontAct,
                                      opponent_class=BaseOpponent,
                                      opponent_budget_class=NeverAttackBudget,
                                      ...  # other arguments pass to the "make" function
                                      )
                            

.. note:: 
  Currently it's not possible to deactivate an opponent once the environment is created.

  If you want this feature, you can comment the issue https://github.com/rte-france/Grid2Op/issues/426


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Opponent
    :members:
    :autosummary:

.. include:: final.rst