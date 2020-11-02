.. currentmodule:: grid2op.gym_compat
.. _openai-gym:

Compatibility with openAI gym
===================================

The gym framework in reinforcement learning is widely used. Starting from version 1.2.0 we improved the
compatibility with this framework.

Before grid2op 1.2.0 only some classes fully implemented the open AI gym interface:

- the :class:`grid2op.Environment` (with methods such as `env.reset`, `env.step` etc.)
- the :class:`grid2op.Agent` (with the `agent.act` etc.)
- the creation of pre defined environments (with `grid2op.make`)


Starting from 1.2.0 we implemented some automatic converters that are able to automatically map
grid2op representation for the action space and the observation space into open AI gym "spaces". More precisely these
are represented as gym.spaces.Dict.

As of grid2op 1.4.0 we tighten the gap between openAI gym and grid2op by introducing the dedicated module
`grid2op.gym_compat` . Withing this module there are lots of functionalities to convert a grid2op environment
into a gym environment (that inherit `gym.Env` instead of "simply" implementing the open ai gym interface).


A simple usage is:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv

    env_name = "l2rpn_case14_sandbox"  # or any other grid2op environment name
    g2op_env = grid2op.make(env_name)  # create the gri2op environment

    gym_env = GymEnv(g2op_env)  # create the gym environment

    # check that this is a properly defined gym environment:
    import gym
    print(f"Is gym_env and open AI gym environment: {isinstance(gym_env, gym.Env)}")
    # it shows "Is gym_env and open AI gym environment: True"

Observation space and action space customization
-------------------------------------------------
By default, the action space and observation space are `gym.spaces.Dict` with the keys being the attribute
to modify.

For example, an observation space, for the environment "l2rpn_case14_sandbox" will look like:

- "a_ex": Box(20,)
- "a_or": Box(20,)
- "actual_dispatch": Box(6,)
- "day": Discrete(32)
- "day_of_week": Discrete(8)
- "duration_next_maintenance": Box(20,)
- "hour_of_day": Discrete(24)
- "line_status": MultiBinary(20)
- "load_p": Box(11,)
- "load_q": Box(11,)
- "load_v": Box(11,)
- "minute_of_hour": Discrete(60)
- "month": Discrete(13)
- "p_ex": Box(20,)
- "p_or": Box(20,)
- "prod_p": Box(6,)
- "prod_q": Box(6,)
- "prod_v": Box(6,)
- "q_ex": Box(20,)
- "q_or": Box(20,)
- "rho": Box(20,)
- "target_dispatch": Box(6,)
- "time_before_cooldown_line": Box(20,)
- "time_before_cooldown_sub": Box(14,)
- "time_next_maintenance": Box(20,)
- "timestep_overflow": Box(20,)
- "topo_vect": Box(57,)
- "v_ex": Box(20,)
- "v_or": Box(20,)
- "year": Discrete(2100)

each keys correspond to an attribute of the observation. In this example `"line_status": MultiBinary(20)`
represents the attribute `obs.line_status` which is a boolean vector (for each powerline
`True` encodes for "connected" and `False` for "disconnected") See the chapter :ref:`observation_module` for
more information

We offer some convenience functions to customize these environment. They all more or less the same
manner. We show here an example of a "converter" that will scale the data:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv
    from grid2op.gym_compat import ScalerAttrConverter

    env_name = "l2rpn_case14_sandbox"  # or any other grid2op environment name
    g2op_env = grid2op.make(env_name)  # create the gri2op environment

    gym_env = GymEnv(g2op_env)  # create the gym environment

    ob_space = gym_env.observation_space
    ob_space = ob_space.reencode_space("actual_dispatch",
                                       ScalerAttrConverter(substract=0.,
                                       divide=env.gen_pmax,
                                       init_space=ob_space["actual_dispatch"])
                                       )

    gym_env.observation_space = ob_space


A detailled list of such "converter" is documented on the section "Detailed Documentation by class". In
the table below we describe some of them (**nb** if you notice a converter is not displayed there,
do not hesitate to write us a "feature request" for the documentation, thanks in advance)

======================================   ============================================================
Converter name                           Objective
======================================   ============================================================
:class:`ContinuousToDiscreteConverter`   TODO
:class:`MultiToTupleConverter`           TODO
:class:`ScalerAttrConverter`             TODO
======================================   ============================================================


Legacy version
---------------

If you are interested by this feature, we recommend you to proceed like this:

.. code-block:: python

   import grid2op
   from grid2op.gym_compat import GymActionSpace, GymObservationSpace
   from grid2op.Agent import BaseAgent

   class MyAgent(BaseAgent):
      def __init__(self, action_space, observation_space):
         BaseAgent.__init__(self, action_space)
         self.gym_obs_space = GymObservationSpace(observation_space)
         self.gym_action_space = GymActionSpace(observation_space)

      def act(self, obs, reward, done=False):
         # convert the observation to gym like one:
         gym_obs = self.gym_obs_space.to_gym(obs)

         # do whatever you want, as long as you retrieve a gym-like action
         gym_action = ...
         grid2op_action = self.gym_action_space.from_gym(gym_action)
         # NB advanced usage: if action_space is a grid2op.converter (for example coming from IdToAct)
         # then what's called  "grid2op_action" is in fact an action that can be understood by the converter.
         # to convert it back to grid2op action you need to convert it. See the documentation of GymActionSpace
         # for such purpose.
         return grid2op_action

   env = grid2op.make(...)
   my_agent = MyAgent(env.action_space, env.observation_space, ...)

   # and now do anything you like
   # for example
   done = False
   reward = env.reward_range[0]
   obs = env.reset()
   while not done:
      action = my_agent.act(obs, reward, done)
      obs, reward, done, info = env.step(action)

We also implemented some "converter" that allow the conversion of some action space into more convenient
`gym.spaces` (this is only available if gym is installed of course). Please check
:class:`grid2op.gym_compat.GymActionSpace` for more information and examples.


Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.gym_compat
    :members:
    :autosummary:

.. include:: final.rst