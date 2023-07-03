
.. _BaseGymSpaceConverter.add_key: ./gym.html#grid2op.gym_compat.gym_space_converter._BaseGymSpaceConverter.add_key
.. _BaseGymSpaceConverter.keep_only_attr: ./gym.html#grid2op.gym_compat.gym_space_converter._BaseGymSpaceConverter.keep_only_attr
.. _BaseGymSpaceConverter.ignore_attr: ./gym.html#grid2op.gym_compat.gym_space_converter._BaseGymSpaceConverter.ignore_attr

.. currentmodule:: grid2op.gym_compat

.. _openai-gym:

Compatibility with gymnasium / gym
===================================

The gymnasium framework in reinforcement learning is widely used. Starting from version 1.2.0 we improved the
compatibility with this framework.

Starting with 1.9.1 we switch (as advised) from the legacy "gym" framework to the 
new "gymnasium" framework (gym is no longer maintained since v0.26.2, see 
https://www.gymlibrary.dev/). This change should not have any impact on older grid2op code
except that you now need to use `import gymnasium as gym` instead of `import gym` in 
your base code.

.. note::
    If you want to still use the "legacy" gym classes you can still do it with grid2op:
    Backward compatibility with openai gym is maintained.

.. note::
    By default, if gymnasium is installed, all default classes from `grid2op.gym_compat` module will 
    inherit from gymnasium. You can still retrieve the classes inheriting from gym (and not gymnasium).

    More information on the section :ref:`gymnasium_gym`

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

.. note::

    To be as close as grid2op as possible, by default (using the methode discribed above) the action
    space will be encoded as a gymnasium.spaces.Dict with keys the attribute of a grid2op action. This might not
    be the best representation to perform RL with (some framework do not really like it...)

    For more customization on that side, please refer to the section :ref:`gym_compat_box_discrete` below

.. warning::
    The `gym` package has some breaking API change since its version 0.26. We attempted, 
    in grid2op, to maintain compatibility both with former versions and later ones. This makes **this
    class behave differently depending on the version of gym you have installed** !
    
    The main changes involve the functions `env.step` and `env.reset` (core gym functions)
    
This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Observation space and action space customization
-------------------------------------------------
By default, the action space and observation space are `gym.spaces.Dict` with the keys being the attribute
to modify.

Default Observations space
******************************
For example, an observation space will look like:

- "_shunt_p": Box(`env.n_shunt`,) [type: float, low: -inf, high: inf]
- "_shunt_q": Box(`env.n_shunt`,) [type: float, low: -inf, high: inf]
- "_shunt_v": Box(`env.n_shunt`,) [type: float, low: -inf, high: inf]
- "_shunt_bus": Box(`env.n_shunt`,) [type: int, low: -inf, high: inf]
- "a_ex": Box(`env.n_line`,) [type: float, low: 0, high: inf]
- "a_or": Box(`env.n_line`,) [type: float, low: 0, high: inf]
- "actual_dispatch": Box(`env.n_gen`,)
- "attention_budget": Box(1,) [type: float, low: 0, high: inf]
- "current_step": Box(1,) [type: int, low: -inf, high: inf]
- "curtailment": Box(`env.n_gen`,)  [type: float, low: 0., high: 1.0]
- "curtailment_limit": Box(`env.n_gen`,)  [type: float, low: 0., high: 1.0]
- "curtailment_limit_effective": Box(`env.n_gen`,)  [type: float, low: 0., high: 1.0]
- "day": Discrete(32)
- "day_of_week": Discrete(8)
- "delta_time": Box(0.0, inf, (1,), float32)
- "duration_next_maintenance": Box(`env.n_line`,)  [type: int, low: -1, high: inf]
- "gen_p": Box(`env.n_gen`,)  [type: float, low: `env.gen_pmin`, high: `env.gen_pmax * 1.2`]
- "gen_p_before_curtail": Box(`env.n_gen`,)  [type: float, low: `env.gen_pmin`, high: `env.gen_pmax * 1.2`]
- "gen_q": Box(`env.n_gen`,)  [type: float, low: -inf, high: inf]
- "gen_v": Box(`env.n_gen`,)  [type: float, low: 0, high: inf]
- "gen_margin_up": Box(`env.n_gen`,)  [type: float, low: 0, high: `env.gen_max_ramp_up`]
- "gen_margin_down": Box(`env.n_gen`,)  [type: float, low: 0, high: `env.gen_max_ramp_down`]
- "hour_of_day": Discrete(24)
- "is_alarm_illegal": Discrete(2)
- "line_status": MultiBinary(`env.n_line`)
- "load_p": Box(`env.n_load`,) [type: float, low: -inf, high: inf]
- "load_q": Box(`env.n_load`,) [type: float, low: -inf, high: inf]
- "load_v": Box(`env.n_load`,) [type: float, low: -inf, high: inf]
- "max_step": Box(1,) [type: int, low: -inf, high: inf]
- "minute_of_hour": Discrete(60)
- "month": Discrete(13)
- "p_ex": Box(`env.n_line`,)  [type: float, low: -inf, high: inf]
- "p_or": Box(`env.n_line`,)  [type: float, low: -inf, high: inf]
- "q_ex": Box(`env.n_line`,)  [type: float, low: -inf, high: inf]
- "q_or": Box(`env.n_line`,)  [type: float, low: -inf, high: inf]
- "rho": Box(`env.n_line`,)  [type: float, low: 0., high: inf]
- "storage_charge": Box(`env.n_storage`,)  [type: float, low: 0., high: `env.storage_Emax`]
- "storage_power": Box(`env.n_storage`,)  [type: float, low: `-env.storage_max_p_prod`, high: `env.storage_max_p_absorb`]
- "storage_power_target": Box(`env.n_storage`,)  [type: float, low: `-env.storage_max_p_prod`, high: `env.storage_max_p_absorb`]
- "target_dispatch": Box(`env.n_gen`,)
- "theta_or": Box(`env.n_line`,)  [type: float, low: -180., high: 180.]
- "theta_ex": Box(`env.n_line`,)  [type: float, low: -180., high: 180.]
- "load_theta": Box(`env.n_load`,)  [type: float, low: -180., high: 180.]
- "gen_theta": Box(`env.n_gen`,)  [type: float, low: -180., high: 180.]
- "storage_theta": : Box(`env.n_storage`,)  [type: float, low: -180., high: 180.]
- "time_before_cooldown_line": Box(`env.n_line`,) [type: int, low: 0, high: depending on parameters]
- "time_before_cooldown_sub": Box(`env.n_sub`,)  [type: int, low: 0, high: depending on parameters]
- "time_next_maintenance": Box(`env.n_line`,)  [type: int, low: 0, high: inf]
- "time_since_last_alarm": Box(1,)  [type: int, low: -1, high: inf]
- "timestep_overflow": Box(`env.n_line`,)  [type: int, low: 0, high: inf]
- "thermal_limit": Box(`env.n_line`,)  [type: int, low: 0, high: inf]
- "topo_vect": Box(`env.dim_topo`,)  [type: int, low: -1, high: 2]
- "v_ex": Box(`env.n_line`,)  [type: float, low: 0, high: inf]
- "v_or": Box(`env.n_line`,)  [type: flaot, low: 0, high: inf]
- "was_alarm_used_after_game_over": Discrete(2)
- "year": Discrete(2100)

Each keys correspond to an attribute of the observation. In this example `"line_status": MultiBinary(20)`
represents the attribute `obs.line_status` which is a boolean vector (for each powerline
`True` encodes for "connected" and `False` for "disconnected") See the chapter :ref:`observation_module` for
more information about these attributes.

You can transform the observation space as you wish. There are some examples in the notebooks.

Default Action space
******************************
The default action space is also a type of gym Dict. As for the observation space above, it is a
straight translation from the attribute of the action to the key of the dictionary. This gives:

- "change_bus": MultiBinary(`env.dim_topo`)
- "change_line_status": MultiBinary(`env.n_line`)
- "curtail": Box(`env.n_gen`) [type: float, low=0., high=1.0]
- "redispatch": Box(`env.n_gen`) [type: float, low=-`env.gen_max_ramp_down`, high=`env.gen_max_ramp_up`]
- "set_bus": Box(`env.dim_topo`) [type: int, low=-1, high=2]
- "set_line_status": Box(`env.n_line`) [type: int, low=-1, high=1]
- "storage_power": Box(`env.n_storage`) [type: float, low=-`env.storage_max_p_prod`, high=`env.storage_max_p_absorb`]

For example you can create a "gym action" (for the default encoding) like:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv
    import numpy as np

    env_name = ...

    env = grid2op.make(env_name)
    gym_env = GymEnv(env)

    seed = ...
    obs, info = gym_env.reset(seed)  # for new gym interface

    # do nothing
    gym_act = {}
    obs, reward, done, truncated, info = gym_env.step(gym_act)

    #change the bus of the element 6 and 7 of the "topo_vect"
    gym_act = {}
    gym_act["change_bus"] = np.zeros(env.dim_topo, dtype=np.int8)   # gym encoding of a multi binary
    gym_act["change_bus"][[6, 7]] = 1
    obs, reward, done, truncated, info = gym_env.step(gym_act)

    # redispatch generator 2 of 1.7MW
    gym_act = {}
    gym_act["redispatch"] = np.zeros(env.n_gen, dtype=np.float32)   # gym encoding of a Box
    gym_act["redispatch"][2] = 1.7
    obs, reward, done, truncated, info = gym_env.step(gym_act)

    # set the bus of element 8 and 9 to bus 2
    gym_act = {}
    gym_act["set_bus"] = np.zeros(env.dim_topo, dtype=int)   # gym encoding of a Box
    gym_act["set_bus"][[8, 9]] = 2
    obs, reward, done, truncated, info = gym_env.step(gym_act)

    # of course, you can set_bus, redispatch, change the storage units etc. in the same action.


This way of doing things is perfectly grounded. It works but it is quite verbose and not
really "ML friendly". You can customize the way you "encode" your actions / observations relatively
easily. Some examples are given in the following subsections.

.. _base_gym_space_function:

Customizing the action and observation space
********************************************

We offer some convenience functions to customize these spaces.

If you want a full control on this spaces, you need to implement something like:

.. code-block:: python

    import grid2op
    env_name = ...
    env = grid2op.make(env_name)

    from grid2op.gym_compat import GymEnv
    # this of course will not work... Replace "AGymSpace" with a normal gym space, like Dict, Box, MultiDiscrete etc.
    from gym.spaces import AGymSpace
    gym_env = GymEnv(env)

    class MyCustomObservationSpace(AGymSpace):
        def __init__(self, whatever, you, want):
            # do as you please here
            pass
            # don't forget to initialize the base class
            AGymSpace.__init__(self, see, gym, doc, as, to, how, to, initialize, it)
            # eg. Box.__init__(self, low=..., high=..., dtype=float)

        def to_gym(self, observation):
            # this is this very same function that you need to implement
            # it should have this exact name, take only one observation (grid2op) as input
            # and return a gym object that belong to your space "AGymSpace"
            return SomethingThatBelongTo_AGymSpace
            # eg. return np.concatenate((obs.gen_p * 0.1, np.sqrt(obs.load_p))

    gym_env.observation_space = MyCustomObservationSpace(whatever, you, wanted)

And for the action space:

.. code-block:: python

    import grid2op
    env_name = ...
    env = grid2op.make(env_name)

    from grid2op.gym_compat import GymEnv
    # this of course will not work... Replace "AGymSpace" with a normal gym space, like Dict, Box, MultiDiscrete etc.
    from gym.spaces import AGymSpace
    gym_env = GymEnv(env)

    class MyCustomActionSpace(AGymSpace):
        def __init__(self, whatever, you, want):
            # do as you please here
            pass
            # don't forget to initialize the base class
            AGymSpace.__init__(self, see, gym, doc, as, to, how, to, initialize, it)
            # eg. MultiDiscrete.__init__(self, nvec=...)

        def from_gym(self, gym_action):
            # this is this very same function that you need to implement
            # it should have this exact name, take only one action (member of your gym space) as input
            # and return a grid2op action
            return TheGymAction_ConvertedTo_Grid2op_Action
            # eg. return np.concatenate((obs.gen_p * 0.1, np.sqrt(obs.load_p))

    gym_env.action_space = MyCustomActionSpace(whatever, you, wanted)

There are some pre defined transformation (for example transforming the action to Discrete or MultiDiscrete). 
Do not hesitate to have a look at the  section :ref:`gym_compat_box_discrete`.


Some already implemented customization
***************************************
However, if you don't want to fully customize everything, we encourage you to have a look at the "GymConverter"
that we coded to ease this process.

They all more or less the same
manner. We show here an example of a "converter" that will scale the data (removing the value in `substract`
and divide input data by `divide`):

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
                                                           init_space=ob_space["actual_dispatch"]
                                                           )
                                       )

    gym_env.observation_space = ob_space


You can also add a specific keys into this observation space, for example say you want to compute
the log of the loads instead of giving the direct value to your agent. This can be done with:

.. code-block:: python

    import grid2op
    from grid2op.gym_compat import GymEnv
    from grid2op.gym_compat import ScalerAttrConverter

    env_name = "l2rpn_case14_sandbox"  # or any other grid2op environment name
    g2op_env = grid2op.make(env_name)  # create the gri2op environment

    gym_env = GymEnv(g2op_env)  # create the gym environment

    ob_space = gym_env.observation_space
    shape_ = (g2op_env.n_load, )
    ob_space = ob_space.add_key("log_load",
                                 lambda obs: np.log(obs.load_p),
                                          Box(shape=shape_,
                                              low=np.full(shape_, fill_value=-np.inf, dtype=float),
                                              high=np.full(shape_, fill_value=-np.inf, dtype=float),
                                              dtype=float
                                              )
                                       )

    gym_env.observation_space = ob_space
    # and now you will get the key "log_load" as part of your gym observation.

A detailed list of such "converter" is documented on the section "Detailed Documentation by class". In
the table below we describe some of them (**nb** if you notice a converter is not displayed there,
do not hesitate to write us a "feature request" for the documentation, thanks in advance)

=============================================   ============================================================
Converter name                                  Objective
=============================================   ============================================================
:class:`ContinuousToDiscreteConverter`          Convert a continuous space into a discrete one
:class:`MultiToTupleConverter`                  Convert a gym MultiBinary to a gym Tuple of gym Binary and a gym MultiDiscrete to a Tuple of Discrete
:class:`ScalerAttrConverter`                    Allows to scale (divide an attribute by something and subtract something from it)
`BaseGymSpaceConverter.add_key`_                Allows you to compute another "part" of the observation space (you add an information to the gym space)
`BaseGymSpaceConverter.keep_only_attr`_         Allows you to specify which part of the action / observation you want to keep
`BaseGymSpaceConverter.ignore_attr`_            Allows you to ignore some attributes of the action / observation (they will not be part of the gym space)
=============================================   ============================================================

.. warning::
    TODO: Help more than welcome !

    Organize this page with a section for each "use":
    
    - scale de the data
    - keep only some part of the observation
    - add some info to the observation
    - transform a box to a discrete action space
    - use MultiDiscrete

    Instead of having the current ordering of things
    
.. note::

    With the "converters" above, note that the observation space AND action space will still
    inherit from gym Dict.

    They are complex spaces that are not well handled by some RL framework.

    These converters only change the keys of these dictionaries !


.. _gym_compat_box_discrete:

Customizing the action and observation space, into Box or Discrete
*******************************************************************

The use of the converter above is nice if you can work with gym Dict, but in some cases, or for some frameworks
it is not convenient to do it at all.

TO alleviate this problem, we developed 3 types of gym action space, following the architecture
detailed in subsection :ref:`base_gym_space_function`

===============================   ============================================================
Converter name                    Objective
===============================   ============================================================
:class:`BoxGymObsSpace`           Convert the observation space to a single "Box"
:class:`BoxGymActSpace`           Convert a gym MultiBinary to a gym Tuple of gym Binary and a gym MultiDiscrete to a Tuple of Discrete
:class:`MultiDiscreteActSpace`    Allows to scale (divide an attribute by something and subtract something from it)
:class:`DiscreteActSpace`         Allows you to compute another "part" of the observation space (you add an information to the gym space)
===============================   ============================================================

They can all be used like:

.. code-block:: python

    import grid2op
    env_name = ...
    env = grid2op.make(env_name)

    from grid2op.gym_compat import GymEnv, BoxGymObsSpace
    gym_env = GymEnv(env)
    gym_env.observation_space = BoxGymObsSpace(gym_env.init_env)
    gym_env.action_space = MultiDiscreteActSpace(gym_env.init_env)


We encourage you to visit the documentation for more information on how to use these classes. Each offer
different possible customization.

.. _gymnasium_gym:

Gymnasium vs Gym
------------------

Starting from grid2op 1.9.1 we introduced the compatibility with `gymnasium` package (the replacement of the
`gym` package that will no longer be maintained).

By default, if gymnasium is installed on your machine, all classes from the `grid2op.gym_compat` module will inherit
from gymnasium. That is :class:`GymEnv` will be inherit from `gymnasium.Env`(and not `gym.Env`), :class:`GymActionSpace`
will inherit from `gymnasium.spaces.Dict` (and not from `gym.spaces.Dict`) etc.

But we wanted to maintain Backward compatibility. It is ensured in two different ways:

1) if you have both `gymnasium` and `gym` installed on your machine, you can choose which "framework"
   you want to use by explicitly using the right grid2op class. For example, if you want a `gym` 
   environment (inheriting from `gym.Env`) you can use :class:`GymEnv_Modern`and if you 
   want to explicitly stay in `gymnasium` you can use :class:`GymnasiumEnv`
2) if you don't want to have `gymnasium` and only `gym` is installed then the default
   grid2op class will stay in the `gym` eco system. In this case, `gym.Env` will
   be :class:`GymEnv_Modern` and all the code previously written will work exactly as
   before.


.. note::
    As you understood if you want to keep the behaviour of grid2op prior to 1.9.1 the simplest solution would be 
    not to install gymnasium at all.

    If however you want to benefit from the latest gymnasium package, you can keep the previous code you have and
    simply install gymnasium. All classes defined there will still be defined and you will be able
    to use gymnasium transparently.

The table bellow summarize the correspondance between the default classes and the classes specific to gymnasium / gym:

======================================  ===============================================  =====================================================
Default class                           Class with gymnasium                             Class with gym
======================================  ===============================================  =====================================================
:class:`BaseGymAttrConverter`           :class:`BaseGymnasiumAttrConverter`              :class:`BaseLegacyGymAttrConverter`
:class:`BoxGymActSpace`                 :class:`BoxGymnasiumActSpace`                    :class:`BoxLegacyGymActSpace`
:class:`BoxGymObsSpace`                 :class:`BoxGymnasiumObsSpace`                    :class:`BoxLegacyGymObsSpace`
:class:`ContinuousToDiscreteConverter`  :class:`ContinuousToDiscreteConverterGymnasium`  :class:`ContinuousToDiscreteConverterLegacyGym`
:class:`DiscreteActSpace`               :class:`DiscreteActSpaceGymnasium`               :class:`DiscreteActSpaceLegacyGym`
:class:`GymActionSpace`                 :class:`GymnasiumActionSpace`                    :class:`LegacyGymActionSpace`
:class:`GymObservationSpace`            :class:`GymnasiumObservationSpace`               :class:`LegacyGymObservationSpace`
:class:`_BaseGymSpaceConverter`         :class:`_BaseGymnasiumSpaceConverter`            :class:`_BaseLegacyGymSpaceConverter`
:class:`GymEnv`                         :class:`GymnasiumEnv`                            :class:`GymEnv_Modern` / :class:`GymEnv_Legacy`
:class:`MultiToTupleConverter`          :class:`MultiToTupleConverterGymnasium`          :class:`MultiToTupleConverterLegacyGym`
:class:`MultiDiscreteActSpace`          :class:`MultiDiscreteActSpaceGymnasium`          :class:`MultiDiscreteActSpaceLegacyGym`
:class:`ScalerAttrConverter`            :class:`ScalerAttrConverterGymnasium`            :class:`ScalerAttrConverterLegacyGym`
======================================  ===============================================  =====================================================


Recommended usage of grid2op with other framework
--------------------------------------------------

Reinforcement learning frameworks
*********************************

TODO

Any contribution is welcome here

Other frameworks
**********************
Any contribution is welcome here too (-:

Troubleshoot with some frameworks
-------------------------------------------------

Python complains about pickle
********************************
This usually takes the form of an error with `XXX_env_name` (*eg* `CompleteObservation_l2rpn_wcci_2022`) is not serializable.

This is because grid2op will (to save computation time) generate some classes (the classes themseleves) on the
fly, once the environment is loaded. And unfortunately, pickle module is not always able to process these
(meta) data.

Try to first create (automatically!) the files containing the description of the classes 
used by your environment (for example):

.. code-block:: python

    from grid2op import make
    from grid2op.Reward import RedispReward
    from lightsim2grid import LightSimBackend

    env_name = 'l2rpn_wcci_2022'
    backend_class = LightSimBackend
    env = make(env_name, reward_class=RedispReward, backend=backend_class())
    env.generate_classes()

.. note::
    This piece of code is to do once (each time you change the backend or the env name)

And then proceed as usual by loading the grid2op environment
with the key-word `experimental_read_from_local_dir`

.. code-block:: python

    from grid2op import make
    from grid2op.Reward import RedispReward
    from lightsim2grid import LightSimBackend

    env_name = 'l2rpn_wcci_2022'
    backend_class = LightSimBackend
    env = make(env_name, reward_class=RedispReward, backend=backend_class(),
               experimental_read_from_local_dir=True)
    # do whatever

Observation XXX outside given space YYY
****************************************
Often encountered with ray[rllib] this is due to a technical aspect (slack bus) of the grid
which may cause issue with gen_p being above / bellow pmin / pmax for certain generators.

You can get rid of it by modifying the observation space and "remove" the low / high values on 
pmin and pmax: 

.. code-block:: python

        # we suppose you already have an observation space
        self.observation_space["gen_p"].low[:] = -np.inf
        self.observation_space["gen_p"].high[:] = np.inf

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.gym_compat
    :members:
    :autosummary:

.. autoclass:: grid2op.gym_compat.gym_space_converter._BaseGymSpaceConverter
    :members:
    :autosummary:

.. include:: final.rst