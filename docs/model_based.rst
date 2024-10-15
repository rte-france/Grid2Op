.. _model_based_rl:

Model Based / Planning methods
====================================


This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
----------------

.. warning::
    This page is in progress. We welcome any contribution :-)

There are 3 standard methods currently in grid2op to apply "model based" / "planning" methods:

1) use "obs.simulate" (see :func:`grid2op.Observation.BaseObservation.simulate`)
2) use the "Simulator" (see :ref:`simulator_page` and :mod:`grid2op.simulator`)
3) use the "forecast env" (see :func:`grid2op.Observation.BaseObservation.get_forecast_env`)
4) use the "forecast env" (see :func:`grid2op.Observation.BaseObservation.get_env_from_external_forecasts`)

.. note::
    The main difference between :func:`grid2op.Observation.BaseObservation.get_forecast_env` 
    and :func:`grid2op.Observation.BaseObservation.get_env_from_external_forecasts`
    is that the first one rely on provided forecast in the environment
    and in :func:`grid2op.Observation.BaseObservation.get_env_from_external_forecasts`
    you are responsible for providing these forecasts.

    This has some implications: 

    - you cannot use `obs.get_forecast_env()` if the forecasts are deactivated,
      or if there are no provided forecast in the environment
    - the number of steps possible in `obs.get_forecast_env()` is fixed and determined
      by the environment.
    - `"garbarge in" = "garbage out"` is especially true for `obs.get_env_from_external_forecasts`
      By this I mean that if you provided forecasts with poor quality (*eg* that does 
      not contain any usefull information about the future, or such that the total generation is 
      lower that the total demand etc.) then you will most likely not get any usefull information
      from their usage.

And you can use them for different strategies among:

- *Decide when to act or not*: A successful techniques is "do nothing" or to "get back to a reference configuration" when the grid is safe. 
  And it's only when the grid is declared "not safe" that an action is taken. You can declare a grid is
  safe is you can "do nothing" withtout overload for a certain number of steps, or test if there are still no overload even if
  the grid is "under stress" (disconnected line by the opponent, more loads / renewables etc.)
- *Chose the best actions among a short list*: in this usecase you have a short list of actions (hard coded, given by a heuristic, 
  by domain knowledge or by a neural network, etc.)


.. _mb_simulate:

obs.simulate
-------------
The idea here is to "simulate" the impact of an action on "future" grid state(s) before taking this action "for real".

You can use it , for example to select the "best action among *k*" (the *k* actions you selected can come from the output of a
neural net and you take the *k* actions with the highest q-value for example).

In this first example you "simulate" the grid state after having taken your actions for the next 3 steps, and take the 
action with the best "score".

.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent1(BaseAgent):
        def act(self, observation, reward, done=False):
            k_actions = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_score = -99999999
            for act in k_actions:
                _, sim_reward1, done, info = obs.simulate(act, time_step=1)
                _, sim_reward2, done, info = obs.simulate(act, time_step=2)  # if supported by the environment
                _, sim_reward3, done, info = obs.simulate(act, time_step=3)  # if supported by the environment
                this_score = function_to_combine_rewards(sim_reward1, sim_reward2, sim_reward3)
                # select the action with the best score
                if this_score > highest_score:
                    res = act
                    highest_score = this_score
            return res


You can also use it to select the action that keep the grid in a "correct" state for the longest

.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent2(BaseAgent):
        def act(self, observation, reward, done=False):
            k_actions = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_score = -1
            for act in k_actions:
                done = False
                ts_survived = 0
                sim_obs, sim_r, sim_done, sim_info = obs.simulate(act)

                if not sim_done:
                    # you can then start to see how long your survive
                    while not done:
                        ts_survived += 1
                        sim_obs, sim_reward, done, info = sim_obs.simulate(self.action_space())

                # select the action with the best score
                if ts_survived > highest_score:
                    res = act
                    highest_score = ts_survived
            return res

.. note::
    In both cases above, you can evaluate the impact of an entire "strategy" (*here* encoded as "a list of actions" -- 
    the most simple one being "do an action then do nothing as long as you can") if you chain
    the calls to simulate. This would give, for the example 1:

    .. code-block:: python

        from grid2op.Agent import BaseAgent

        class ExampleAgent1Bis(BaseAgent):
            def act(self, observation, reward, done=False):
                k_strategies = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
                res = None
                highest_score = -99999999
                for strat in k_strategies:
                    act1, act2, act3 = strat
                    s_o1, sim_reward1, done, info = obs.simulate(act1)
                    sim_reward2 = None
                    sim_reward3 = None
                    if not done:
                        s_o2, sim_reward2, done, info = s_o1.simulate(act2)
                        if not done:
                            s_o3, sim_reward3, done, info = s_o2.simulate(act3)
                    
                    this_score = function_to_combine_rewards(sim_reward1, sim_reward2, sim_reward3)
                    # select the action with the best score
                    if this_score > highest_score:
                        res = strat[0]  # action will be the first one of the strategy of course
                        highest_score = this_score
            return res

    And for the ExampleAgent2:

    .. code-block:: python

        from grid2op.Agent import BaseAgent

        class ExampleAgent2Bis(BaseAgent):
            def act(self, observation, reward, done=False):
                k_strategies = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
                res = None
                highest_score = -1
                for strat in k_strategies:
                    done = False
                    ts_survived = 0
                    sim_obs, sim_r, sim_done, sim_info = obs.simulate(strat[ts_survived])

                    if not sim_done:
                        # you can then start to see how long your survive
                        while not done:
                            ts_survived += 1
                            sim_obs, sim_reward, done, info = sim_obs.simulate(strat[ts_survived])
                    
                    # select the action with the best score
                    if ts_survived > highest_score:
                        res = strat[0]  # action is the first one of the best strategy
                        highest_score = ts_survived
                return res


.. note::
    We are sure there are lots of other ways to use "obs.simulate". If you have some idea let us know, for example by starting
    a conversation here https://github.com/Grid2Op/grid2op/discussions or in our discord.


Simulator
--------------

The idea of the :class:`grid2op.simulator.Simulator` is to allow you to have more control on the "grid state" you want to simulate.
Instead of relying on pre computed "time series" of the environment (so called "*forecast*") you can define your own "load" and 
"generation".

This can be usefull if you want to see if an action is still persistent if the grid is "more stressed".

In a simular setting that above, you could select the "best action" among a list of *k* based on the "more robust action if the 
loads increase" (there are lots of ways to "stress" a powergrid... You can increase the amount of renewables, the total demand, 
you can increase the demand in a particular area, disconnect some powerlines etc. etc. We keep it simple here and will just increase
the demand - and the generation, because remember that `sum generation = sum load + losses` by a certain factor). 

This can give a code looking like:


.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent3(BaseAgent):
        def act(self, observation, reward, done=False):
            k_actions = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_stress = -1
            simulator = obs.get_simulator()
            
            init_load_p = 1. * obs.load_p
            init_load_q = 1. * obs.load_q
            init_gen_p = 1. * obs.gen_p

            for act in k_actions:
                done = False
                max_stress = 0
                sim_obs, sim_r, sim_done, sim_info = obs.simulate(act)

                # you can stress the grid the way you want, disconnecting some powerline
                # increase demand / generation in certain area etc. etc.
                # we just do a simple "heuristic" here
                for this_stress in [1, 1.01, 1.02, 1.03, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]:
                    this_load_p = init_load_p * this_stress
                    this_load_q = init_load_q * this_stress
                    this_gen_p = init_gen_p * this_stress
                    res = simulator.predict(act,
                                            new_gen_p=new_gen_p,
                                            new_load_p=new_load_p,
                                            new_load_q=new_load_q,
                                            )
                    if not res.converged:
                        # simulation could not be made, this would corresponds to a "game over"
                        break
                    obs = res.current_obs
                    if np.any(obs.rho > 1.):
                        # grid is not safe, action is not "robust enough":
                        # at least one powerline is overloaded
                        break
                    prev_stress = this_stress

                # select the action with the best score
                if prev_stress > highest_stress:
                    res = act
                    highest_stress = prev_stress
            return res

This way of looking at the problem is related to the "forecast error". If you "stress" the grid in the direction where you 
expect the forecast to be inaccurate and you want to know if your "strategy" is robust to these uncertainties.

If you rather want to disconnect some powerline as way to stress the grid, you can end up with something like:

.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent3Bis(BaseAgent):
        def act(self, observation, reward, done=False):
            k_strategies = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_stress = -1
            simulator = obs.get_simulator()

            for act in k_strategies:
                done = False
                this_stress_pass = 0
                sim_obs, sim_r, sim_done, sim_info = obs.simulate(act)

                # you can stress the grid the way you want, disconnecting some powerline
                # increase demand / generation in certain area etc. etc.
                # here we simulate the impact of your action after disconnection of line 1,2, 7, 12 and 42
                for this_stress_id in [1, 2, 7, 12, 42]:
                    this_act = act.copy()
                    this_act += self.action_space({"set_line_status": [(this_stress_id, -1)]})

                    # some code that ignores the "topology" ways (if any) to reconnect the line
                    # in the original action
                    this_act.remove_line_status_from_topo(check_cooldown=False)
                    res = simulator.predict(this_act,
                                            new_gen_p=new_gen_p,
                                            new_load_p=new_load_p,
                                            new_load_q=new_load_q,
                                            )
                    if not res.converged:
                        # simulation could not be made, this would corresponds to a "game over"
                        continue
                    obs = res.current_obs
                    if np.any(obs.rho > 1.):
                        # grid is not safe, action is not "robust enough":
                        # at least one powerline is overloaded
                        continue
                    this_stress_pass += 1

                # select the action with the best score
                # in this case the highest number of "safe disconnection"
                if this_stress_pass > highest_stress:
                    res = act
                    highest_stress = this_stress_pass
            return res


.. note::
    We are sure there are lots of other ways to use "obs.simulate". If you have some idea let us know, for example by starting
    a conversation here https://github.com/Grid2Op/grid2op/discussions or in our discord.


Forecast env
---------------

Finally you can use the :func:`grid2op.Observation.BaseObservation.get_forecast_env` to retrieve an actual
environment already loaded with the "forecast" data available. Alternatively,
if you want to use this feature but the environment does not provide such forecasts
you can have a look at the 
:func:`grid2op.Observation.BaseObservation.get_env_from_external_forecasts` 
(if you can generate your own forecasts) or
the :ref:`tshandler-module` section of the documentation (to still be able
to "generate" forecasts)

Lots of example can be use in this setting, for example using MCTS or any other "planning strategy", but if we take
again the example of the section :ref:`mb_simulate` above this also allows to evaluate the impact of
more than 1 action already planned, or of an action followed by "do nothing" etc.

This could give, for the `ExampleAgent1`

.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent4(BaseAgent):
        def act(self, observation, reward, done=False):
            k_strategies = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_score = -99999999
            for strat in k_strategies:
                act1, act2, act3 = strat
                f_env = obs.get_forecast_env()
                f_obs = f_env.reset()
                done = False
                ts_survived = 0
                strat_rewards = []
                while not done:
                    f_obs, f_r, done, f_info = f_env.step(strat[ts_survived])
                    strat_rewards.append(f_r)
                    ts_survived += 1

                this_score = function_to_combine_rewards(strat_rewards)
                # select the strategy with the best score
                if this_score > highest_score:
                    res = strat[0]  # action will be the first one of the strategy of course
                    highest_score = this_score

            return res

And for the `ExampleAgent2`:

.. code-block:: python

    from grid2op.Agent import BaseAgent

    class ExampleAgent5(BaseAgent):
        def act(self, observation, reward, done=False):
            k_strategies = ...  # whatever you want, hard coded, heuristics, output of a NN etc.
            res = None
            highest_score = -1
            for strat in k_strategies:
                f_done = False
                f_env = obs.get_forecast_env()
                f_obs = f_env.reset()

                ts_survived = 0
                f_obs, f_r, f_done, f_info = f_env.step(strat[ts_survived])

                if not f_done:
                    # you can then start to see how long your survive
                    while not f_done:
                        ts_survived += 1
                        f_obs, f_reward, f_done, f_info = f_env.step(strat[ts_survived])
                
                # select the action with the best score
                if ts_survived > highest_score:
                    res = strat[0]  # action is the first one of the best strategy
                    highest_score = ts_survived
            return res

.. include:: final.rst
