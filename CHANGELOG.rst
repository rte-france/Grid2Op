Change Log
===========

[TODO]
--------------------
- [???] closer integration with `gym` especially the "register env", being able to 
  create an env from a string etc.
- [???] clean the notebook on RL
- [???] use the typing module for type annotation.
- [???] use some kind of "env.get_state()" when simulating instead of recoding everything "by hand"
- [???] use "backend.get_action_to_set()" in simulate
- [???] model better the voltage, include voltage constraints
- [???] use the prod_p_forecasted and co in the "next_chronics" of simulate
- [???] add a "_cst_" or something in the `const` member of all the classes
- [???] in deepcopy of env, make tests that the "pointers" are properly propagated in the attributes (for example
  `envcpy._game_rules.legal_action` should not be copied when building `envcpy._helper_action_env`)
- [???] add multi agent
- [???] make observation read only / immutable for all its properties (and not just for `prod_p`)
- [???] better logging
- [???] shunts in observation too, for real (but what to do when backend is not shunt compliant to prevent the
  stuff to break)
- [???] model agent acting at different time frame
- [???] model delay in observations
- [???] model delay in action
- [???] Code and test the "load from disk" method
- [???] Make the redispatching data independent from the time step (eg instead of "in MW / step" have it in "MW / h")
  and have grid2op convert it to MW / step
- [???] add a "plot action" method
- [???] in MultiEnv, when some converter of the observations are used, have each child process to compute
  it in parallel and transfer the resulting data.
- [???] "asynch" multienv
- [???] properly model interconnecting powerlines

Work kind of in progress
----------------------------------
- TODO A number of max buses per sub
- TODO in the runner, save multiple times the same sceanrio
- TODO in the gym env, make the action_space and observation_space attribute
  filled automatically (see ray integration, it's boring to have to copy paste...)

Next release
---------------------------------
- numpy 2 compat (need pandapower for that)
- automatic read from local dir also on windows !
- TODO doc for the "new" feature of automatic "experimental_read_from_local_dir"
- TODO bug on maintenance starting at midnight (they are not correctly handled in the observation)
  => cf script test_issue_616
- TODO put the Grid2opEnvWrapper directly in grid2op as GymEnv
- TODO faster gym_compat (especially for DiscreteActSpace and BoxGymObsSpace)
- TODO Notebook for tf_agents
- TODO Notebook for acme
- TODO Notebook using "keras rl" (see https://keras.io/examples/rl/ppo_cartpole/)
- TODO example for MCTS https://github.com/bwfbowen/muax et https://github.com/google-deepmind/mctx
- TODO jax everything that can be: create a simple env based on jax for topology manipulation, without
  redispatching or rules
- TODO backend in jax, maybe ?
- TODO done and truncated properly handled in gym_compat module (when game over
  before the end it's probably truncated and not done) 
- TODO when reset, have an attribute "reset_infos" with some infos about the
  way reset was called.
- TODO ForecastEnv in MaskedEnv ! (and obs.simulate there too !)
- TODO finish the test in automatic_classes
- TODO in multi-mix increase the reset options with the mix the user wants
- TODO L2RPN scores as reward (sum loads after the game over and have it in the final reward)
- TODO on CI: test only gym, only gymnasium and keep current test for both gym and gymnasium
- TODO work on the reward class (see https://github.com/rte-france/Grid2Op/issues/584)


[1.10.4] - 2024-xx-yy
-------------------------


[1.10.3] - 2024-07-12
-------------------------
- [BREAKING] `env.chronics_hander.set_max_iter(xxx)` is now a private function. Use 
  `env.set_max_iter(xxx)` or even better `env.reset(options={"max step": xxx})`. 
  Indeed, `env.chronics_hander.set_max_iter()` will likely have
  no effect at all on your environment.
- [BREAKING] for all the `Handler` (*eg* `CSVForecastHandler`) the method `set_max_iter` is 
  now private (for the same reason as the `env.chronics_handler`). We do not recommend to
  use it (will likely have no effect). Prefer using `env.set_max_iter` instead.
- [BREAKING] now the `runner.run()` method only accept kwargs argument 
  (because  it should always have been like this)
- [BREAKING] to improve pickle support and multi processing capabilities, the attribute
  `gym_env.observation_space._init_env` and `gym_env.observation_space.initial_obs_space`
  have been deleted (for the `Dict` space only, for the other spaces like the `Box` they
  were not present in the first place)
- [BREAKING] in the `GymEnv` class now by default the underlying grid2op environment has no
  forecast anymore in an attempt to make this wrapper faster AND more easily pickle-able. You can
  retrieve the old behaviour by passing `gym_env = GymEnv(grid2op_env, with_forecast=True)`
- [FIXED] a bug in the `MultiFolder` and `MultifolderWithCache` leading to the wrong 
  computation of `max_iter` on some corner cases
- [FIXED] the function `cleanup_action_space()` did not work correctly when the "chronics_hander"
  was not initialized for some classes
- [FIXED] the `_observationClass` attribute of the "observation env" (used for simulate and forecasted env)
  is now an Observation and not an Action.
- [FIXED] a bug when deep copying an "observation environment" (it changes its class)
- [FIXED] issue on `seed` and `MultifolderWithCache` which caused 
  https://github.com/rte-france/Grid2Op/issues/616 
- [FIXED] another issue with the seeding of `MultifolderWithCache`: the seed was not used
  correctly on the cache data when calling `chronics_handler.reset` multiple times without 
  any changes
- [FIXED] `Backend` now properly raise EnvError (grid2op exception) instead of previously 
  `EnvironmentError` (python default exception)
- [FIXED] a bug in `PandaPowerBackend` (missing attribute) causing directly 
  https://github.com/rte-france/Grid2Op/issues/617
- [FIXED] a bug in `Environment`: the thermal limit were used when loading the environment 
  even before the "time series" are applied (and before the user defined thermal limits were set) 
  which could lead to disconnected powerlines even before the initial step (t=0, when time 
  series are loaded)
- [FIXED] an issue with the "max_iter" for `FromNPY` time series generator
- [FIXED] a bug in `MultiMixEnvironment` : a multi-mix could be created even if the underlying 
  powergrids (for each mix) where not the same.
- [FIXED] a bug in `generate_classes` (experimental_read_from_local_dir) with alert data.
- [FIXED] a bug in the `Runner` when using multi processing on macos and windows OS: some non default
  parameters where not propagated in the "child" process (bug in `runner._ger_params`)
- [ADDED] possibility to skip some step when calling `env.reset(..., options={"init ts": ...})`
- [ADDED] possibility to limit the duration of an episode with `env.reset(..., options={"max step": ...})`
- [ADDED] possibility to specify the "reset_options" used in `env.reset` when
  using the runner with `runner.run(..., reset_options=xxx)`
- [ADDED] the argument `mp_context` when building the runner to help pass a multiprocessing context in the
  grid2op `Runner` 
- [ADDED] the time series are now able to regenerate their "random" part 
  even when "cached" thanks to the addition of the `regenerate_with_new_seed` of the 
  `GridValue` class (in public API)
- [ADDED] `MultifolderWithCache` now supports `FromHandlers` time series generator
- [IMPROVED] more consistency in the way the classes are initialized at the creation of an environment
- [IMPROVED] more consistency when an environment is copied (some attributes of the copied env were 
  deep copied incorrectly)
- [IMPROVED] Doc about the runner
- [IMPROVED] the documentation on the `time series` folder.
- [IMPROVED] now the "maintenance from json" (*eg* the `JSONMaintenanceHandler` or the 
  `GridStateFromFileWithForecastsWithMaintenance`) can be customized with the day 
  of the week where the maintenance happens (key `maintenance_day_of_week`)
- [IMPROVED] in case of "`MultiMixEnvironment`" there is now only class generated for 
  all the underlying mixes (instead of having one class per mixes)
- [IMPROVED] the `EpisodeData` have now explicitely a mode where they can be shared accross 
  processes (using `fork` at least), see `ep_data.make_serializable`
- [IMPROVED] chronix2grid tests are now done independantly on the CI


[1.10.2] - 2024-05-27
-------------------------
- [BREAKING] the `runner.run_one_episode` now returns an extra argument (first position): 
  `chron_id, chron_name, cum_reward, timestep, max_ts = runner.run_one_episode()` which 
  is consistant with `runner.run(...)` (previously it returned only 
  `chron_name, cum_reward, timestep, max_ts = runner.run_one_episode()`)
- [BREAKING] the runner now has no `chronics_handler` attribute (`runner.chronics_handler` 
  is not defined)
- [BREAKING] now grid2op forces everything to be connected at busbar 1 if
  `param.IGNORE_INITIAL_STATE_TIME_SERIE == True` (**NOT** the default) and
  no initial state is provided in `env.reset(..., options={"init state": ...})`
- [ADDED] it is now possible to call `change_reward` directly from 
  an observation (no need to do it from the Observation Space)
- [ADDED] method to change the reward from the observation (observation_space
  is not needed anymore): you can use `obs.change_reward`
- [ADDED] a way to automatically set the `experimental_read_from_local_dir` flags 
  (with automatic class creation). For now it is disable by default, but you can 
  activate it transparently (see doc)
- [ADDED] possibility to set the grid to an initial state (using an action) when using the
  "time series" classes. The supported classes are `GridStateFromFile` - and all its derivative, 
  `FromOneEpisodeData`, `FromMultiEpisodeData`, `FromNPY` and `FromHandlers`. The classes `ChangeNothing`
  and `FromChronix2grid` are not supported at the moment.
- [ADDED] an "Handler" (`JSONInitStateHandler`) that can set the grid to an initial state (so as to make
  compatible the `FromHandlers` time series class with this new feature)
- [ADDED] some more type hints in the `GridObject` class 
- [ADDED] Possibility to deactive the support of shunts if subclassing `PandaPowerBackend`
  (and add some basic tests)
- [ADDED] a parameters (`param.IGNORE_INITIAL_STATE_TIME_SERIE`) which defaults to
  `False` that tells the environment whether it should ignore the 
  initial state of the grid provided in the time series.
  By default it is NOT ignored, it is taken into account 
  (for the environment that supports this feature)
- [FIXED] a small issue that could lead to having 
  "redispatching_unit_commitment_availble" flag set even if the redispatching
  data was not loaded correctly
- [FIXED] EducPandaPowerBackend now properly sends numpy array in the class attributes
  (instead of pandas series)
- [FIXED] an issue when loading back data (with `EpisodeData`): when there were no storage units
  on the grid it did not set properly the "storage relevant" class attributes
- [FIXED] a bug in the "gridobj.generate_classes()" function which crashes when no
  grid layout was set
- [FIXED] notebook 5 on loading back data with `EpisodeData`.
- [FIXED] converter between backends (could not handle more than 2 busbars)
- [FIXED] a bug in `BaseMultiProcessEnvironment`: set_filter had no impact
- [FIXED] an issue in the `Runner` (`self.chronics_handler` was sometimes used, sometimes not
  and most of the time incorrectly)
- [FIXED] on `RemoteEnv` class (impact all multi process environment): the kwargs used to build then backend
  where not used which could lead to"wrong" backends being used in the sub processes.
- [FIXED] a bug when the name of the times series and the names of the elements in the backend were 
  different: it was not possible to set `names_chronics_to_grid` correctly when calling `env.make`
- [IMPROVED] documentation about `obs.simulate` to make it clearer the 
  difference between env.step and obs.simulate on some cases
- [IMPROVED] type hints on some methods of `GridObjects`
- [IMPROVED] replace `np.nonzero(arr)` calls with `arr.nonzero()` which could
  save up a bit of computation time.
- [IMPROVED] force class attributes to be numpy arrays of proper types when the 
  classes are initialized from the backend.
- [IMPROVED] some (slight) speed improvments when comparing actions or deep copying objects
- [IMPROVED] the way the "grid2op compat" mode is handled
- [IMPROVED] the coverage of the tests in the "test_basic_env_ls.py" to test more in depth lightsim2grid
  (creation of multiple environments, grid2op compatibility mode)
- [IMPROVED] the function to test the backend interface in case when shunts are not supported 
  (improved test `AAATestBackendAPI.test_01load_grid`)

[1.10.1] - 2024-03-xx
----------------------
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/593
- [FIXED] backward compatibility issues with "oldest" lightsim2grid versions
  (now tested in basic settings)
- [ADDED] a "compact" way to store the data in the Runner
- [IMPROVED] the "`train_val_split`" functions, now more names (for the folders)
  can be used

[1.10.0] - 2024-03-06
----------------------
- [BREAKING] the order of the actions in `env.action_space.get_all_unitary_line_set` and 
  `env.action_space.get_all_unitary_topologies_set` might have changed (this is caused 
  by a rewriting of these functions in case there is not 2 busbars per substation)
- [FIXED] github CI did not upload the source files
- [FIXED] `l2rpn_utils` module did not stored correctly the order
  of actions and observation for wcci_2020
- [FIXED] 2 bugs detected by static code analysis (thanks sonar cloud)
- [FIXED] a bug in `act.get_gen_modif` (vector of wrong size was used, could lead
  to some crashes if `n_gen >= n_load`)
- [FIXED] a bug in `act.as_dict` when shunts were modified
- [FIXED] a bug affecting shunts: sometimes it was not possible to modify their p / q 
  values for certain values of p or q (an AmbiguousAction exception was raised wrongly)
- [FIXED] a bug in the `_BackendAction`: the "last known topoolgy" was not properly computed
  in some cases (especially at the time where a line was reconnected)
- [FIXED] `MultiDiscreteActSpace` and `DiscreteActSpace` could be the same classes
  on some cases (typo in the code).
- [FIXED] a bug in `MultiDiscreteActSpace` : the "do nothing" action could not be done if `one_sub_set` (or `one_sub_change`)
  was selected in `attr_to_keep`
- [ADDED] a method `gridobj.topo_vect_element()` that does the opposite of `gridobj.xxx_pos_topo_vect`
- [ADDED] a mthod `gridobj.get_powerline_id(sub_id)` that gives the
  id of all powerlines connected to a given substation
- [ADDED] a convenience function `obs.get_back_to_ref_state(...)`
  for the observation and not only the action_space.
- [IMPROVED] handling of "compatibility" grid2op version
  (by calling the relevant things done in the base class 
  in `BaseAction` and `BaseObservation`) and by using the `from packaging import version`
  to check version (instead of comparing strings)
- [IMPROVED] slightly the code of `check_kirchoff` to make it slightly clearer
- [IMRPOVED] typing and doc for some of the main classes of the `Action` module
- [IMRPOVED] typing and doc for some of the main classes of the `Observation` module
- [IMPROVED] methods `gridobj.get_lines_id`, `gridobj.get_generators_id`, `gridobj.get_loads_id`
  `gridobj.get_storages_id` are now class methods and can be used with `type(env).get_lines_id(...)`
  or `act.get_lines_id(...)` for example.
- [IMPROVED] `obs.get_energy_graph()` by giving the "local_bus_id" and the "global_bus_id"
  of the bus that represents each node of this graph.
- [IMPROVED] `obs.get_elements_graph()` by giving access to the bus id (local, global and 
  id of the node) where each element is connected.
- [IMPROVED] description of the different graph of the grid in the documentation.
- [IMPROVED] type hints for the `gym_compat` module (more work still required in this area)
- [IMPROVED] the `MultiDiscreteActSpace` to have one "dimension" controling all powerlines
  (see "one_line_set" and "one_line_change")
- [IMPROVED] doc at different places, including the addition of the MDP implemented by grid2op.

[1.9.8] - 2024-01-26
----------------------
- [FIXED] the `backend.check_kirchoff` function was not correct when some elements were disconnected 
  (the wrong columns of the p_bus and q_bus was set in case of disconnected elements)
- [FIXED] `PandapowerBackend`, when no slack was present
- [FIXED] the "BaseBackendTest" class did not correctly detect divergence in most cases (which lead 
  to weird bugs in failing tests)
- [FIXED] an issue with imageio having deprecated the `fps` kwargs (see https://github.com/rte-france/Grid2Op/issues/569)
- [FIXED] adding the "`loads_charac.csv`" in the package data
- [FIXED] a bug when using grid2op, not "utils.py" script could be used (see 
  https://github.com/rte-france/Grid2Op/issues/577). This was caused by the modification of
  `sys.path` when importing the grid2op test suite.
- [ADDED] A type of environment that does not perform the "emulation of the protections"
  for some part of the grid (`MaskedEnvironment`) see https://github.com/rte-france/Grid2Op/issues/571
- [ADDED] a "gym like" API for reset allowing to set the seed and the time serie id directly when calling
  `env.reset(seed=.., options={"time serie id": ...})`
- [IMPROVED] the CI speed: by not testing every possible numpy version but only most ancient and most recent
- [IMPROVED] Runner now test grid2op version 1.9.6 and 1.9.7
- [IMPROVED] refacto `gridobj_cls._clear_class_attribute` and `gridobj_cls._clear_grid_dependant_class_attributes`
- [IMPROVED] the bahviour of the generic class `MakeBackend` used for the test suite.
- [IMPROVED] re introducing python 12 testing
- [IMPROVED] error messages in the automatic test suite (`AAATestBackendAPI`)

[1.9.7] - 2023-12-01
----------------------
- [BREAKING] removal of the `grid2op/Exceptions/PowerflowExceptions.py` file and move the
  `DivergingPowerflow` as part of the BackendException. If you imported (to be avoided)
  with `from grid2op.Exceptions.PowerflowExceptions import PowerflowExceptions`
  simply do `from grid2op.Exceptions import PowerflowExceptions` and nothing
  will change.
- [BREAKING] rename with filename starting with lowercase all the files in the "`Exceptions`", 
  module. This is both consistent with python practice but allows also to make the 
  difference between the files in the 
  module and the class imported. This should have little to no impact on all codes but to "upgrade"
  instead of `from grid2op.Exceptions.XXX import PowerflowExceptions` (which you should not have done in the first place) 
  just do `from grid2op.Exceptions import PowerflowExceptions`. Expect other changes like this for other grid2op modules
  in the near future.
- [BREAKING] change the `gridobj_cls.shape()` and `gridobj_cls.dtype()` to `gridobj_cls.shapes()` and `gridobj_cls.dtypes()`
  to be more clear when dealing with action_space and observation_space (where `shape` and `dtype` are attribute and not functions)
  This change means you can still use `act.shape()` and `act.dtype()` but that `act_space.shape` and `act_space.dtype` are now
  clearly properties (and NOT attribute). For the old function `gridobj_cls.dtype()` you can now use `gridobj_cls.dtypes()`
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/561 (indent issue)
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/550 : issue with `shunts_data_available` now better handled
- [IMPROVED] the function to check the backend interface now also check that
  the `topo_vect` returns value between 1 and 2.
- [IMPROVED] the function to check backend now also check the `topo_vect`
  for each type of elements.

[1.9.6] - 2023-10-26
----------------------
- [BREAKING] when a storage is connected alone on a bus, even if it produces / absorbs 0.0 MW it 
  will raise a diverging powerflow error (previously the storage was automatically disconnected by 
  `PandaPowerBackend`, but probably not by other backends)
- [BREAKING] when a shunt is alone on a bus, the powerflow will diverge even in DC mode 
  (previously it only converges which was wrong behaviour: grid2op should not disconnect shunt)
- [FIXED] a bug in PandaPowerBackend (DC mode) where isolated load did not raised 
  exception (they should lead to a divergence)
- [FIXED] some wrong behaviour in the `remove_line_status_from_topo` when no observation where provided
  and `check_cooldown` is `False`
- [FIXED] a bug in PandaPowerBackend in AC powerflow: disconnected storage unit had no 0. as voltage
- [FIXED] a bug in PandaPowerBackend in AC powerflow when a generator was alone a bus it made the powerflow
  crash on some cases (*eg* without lightsim2grid, without numba)
- [FIXED] a bug in PandaPowerBackend in DC (in some cases non connected grid were not spotted)
- [FIXED] now the observations once reloaded have the correct `_is_done` flag (`obs._is_done = False`)
  which allows to use the `obs.get_energy_graph()` for example. This fixes https://github.com/rte-france/Grid2Op/issues/538
- [ADDED] now depends on the `typing_extensions` package
- [ADDED] a complete test suite to help people develop new backend using "Test Driven Programming" 
  techniques
- [ADDED] the information on which time series data has been used by the environment in the `info`return value
  of `env.step(...)`
- [ADDED] a test suite easy to set up to test the backend API (and only the backend for now, integration tests with
  runner and environment will follow)
- [ADDED] an attribute of the backend to specify which file extension can be processed by it. Environment creation will
  fail if none are found. See `backend.supported_grid_format` see https://github.com/rte-france/Grid2Op/issues/429
- [IMPROVED] now easier than ever to run the grid2op test suite with a new backend (for relevant tests)
- [IMPROVED] type hints for `Backend` and `PandapowerBackend`
- [IMPROVED] distribute python 3.12 wheel
- [IMPROVED] test for python 3.12 and numpy 1.26 when appropriate (*eg* when numpy version is released)
- [IMPROVED] handling of environments without shunts
- [IMPROVED] error messages when grid is not consistent 
- [IMPROVED] add the default `l2rpn_case14_sandbox` environment in all part of the docs (substituing `rte_case14_realistic` or nothing)
- [IMPROVED] imports on the `Exceptions` module
- [IMPROVED] pandapower backend raises `BackendError` when "diverging"

[1.9.5] - 2023-09-18
---------------------
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/518
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/446
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/523 by having a "_BackendAction" folder instead of a file
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/522 and adding back certain notebooks to the CI
- [FIXED] an issue when disconnecting loads / generators on msot recent pandas version
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/527 : now do nothing action are detected in 
  `act.as_serializable_dict()` AND weird do nothing action can be made through the action space
  (`env.action_space({"change_bus": {}})` is not ambiguous, though might not be super efficient...)

[1.9.4] - 2023-09-04
---------------------
- [FIXED] read-the-docs template is not compatible with latest sphinx version (7.0.0)
  see https://github.com/readthedocs/sphinx_rtd_theme/issues/1463
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/511
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/508
- [ADDED] some classes that can be used to reproduce exactly what happened in a previously run environment
  see `grid2op.Chronics.FromOneEpisodeData` and `grid2op.Opponent.FromEpisodeDataOpponent` 
  and `grid2op.Chronics.FromMultiEpisodeData`
- [ADDED] An helper function to get the kwargs to disable the opponent (see `grid2op.Opponent.get_kwargs_no_opponent()`)
- [IMPROVED] doc of `obs.to_dict` and `obs.to_json` (see https://github.com/rte-france/Grid2Op/issues/509)

[1.9.3] - 2023-07-28
---------------------
- [BREAKING] the "chronix2grid" dependency now points to chronix2grid and not to the right branch
  this might cause an issue if you install `grid2op[chronix2grid]` for the short term
- [BREAKING] force key-word arguments in `grid2op.make` except for the first one (env name), see
  [rte-france#503](https://github.com/rte-france/Grid2Op/issues/503)
- [FIXED] a bug preventing to use storage units in "sim2real" environment (when the 
  grid for forecast is not the same as the grid for the environment)
- [ADDED] a CI to test package can be installed and loaded correctly on windows, macos and line_ex_to_sub_pos
  for python 3.8, 3.9, 3.10 and 3.11
- [ADDED] possibility to change the "soft_overflow_threshold" in the parameters (like
  the "hard_overflow_threshold" but for delayed protections). 
  See `param.SOFT_OVERFLOW_THRESHOLD`
- [ADDED] the `gym_env.observation_space.get_index(attr_nm)` for `BoxGymObsSpace` that allows to retrieve which index
  of the observation represents which attribute.

[1.9.2] - 2023-07-26
---------------------
- [BREAKING] rename with filename starting with lowercase all the files in the "`Backend`", "`Action`" and 
  "`Environment`" modules. This is both consistent with python practice but allows also to make the 
  difference between the files in the 
  module and the class imported. This should have little to no impact on all codes but to "upgrade"
  instead of `from grid2op.Action.BaseAction import BaseAction` (which you should not have done in the first place) 
  just do `from grid2op.Action import BaseAction`. Expect other changes like this for other grid2op modules
  in the near future.
- [FIXED] broken environ "l2rpn_idf_2023" (with test=True) due to the presence of a `__pycache__` folder
- [FIXED] time series `MultiFolder` will now ignore folder `__pycache__`
- [FIXED] an issue with compatibility with previous versions (due to alert)
- [FIXED] an issue with the `_ObsEnv` when using reward that could not be used in forecast (`self.is_simulated_env()`
  was not working as expected due to a wrong init of the reward in `_ObsEnv`)
- [FIXED] an issue when disconnecting loads / generators / storage units and changing their values in the same
  action: the behaviour could depend on the backend. As of 1.9.2 the "disconnections" have the priority  (if 
  an action disconnect an element, it will not change its sepoint at the same time). 
- [FIXED] a bug in `AlertReward` due to `reset` not being called.
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/494
- [ADDED] the score function used for the L2RPN 2023 competition (Paris Area)
- [IMPROVED] overall performances by calling `arr.sum()` or `arr.any()` instead of `np.sum(arr)` or
  `np.any(arr)` see https://numpy.org/neps/nep-0018-array-function-protocol.html#performance
- [IMPROVED] overall performance of `obs.simulate` function by improving speed of copy of `_BackendAction`
- [IMPROVED] overall performance of `env.step` / `obs.simulate` by preventing unnecessary observation deep copy
- [IMPROVED] overall performance of `env.step` / `obs.simulate` by switching to `copy.deepcopy(obs)` instead of
  `obs.copy()`
  
[1.9.1] - 2023-07-06
--------------------
- [BREAKING] (slightly): default `gym_compat` module now inherit from `gymnasium` (if 
  gymnasium is installed) instead of `gym`. If you want legacy behaviour, 
  do not install `gymnasium`. If you want compatibility with sota softwares using `gymnasium`,
  install it and continue using grid2op transparently. See doc of `gym_compat` module for more
  information.
- [BREAKING] remove the support of the "raise_alarm" kwargs in the DiscreteActSpace
- [BREAKING] remove support for python 3.7 that has reached end of life on 2023-06-27 on
  pypi and on CI
- [BREAKING] to avoid misleading behaviour, by default the `BoxGymActSpace` no longer uses
  the "discrete" attributes ("set_line_status", "change_line_status", "set_bus", "change_bus"). You can
  still use them in the "attr_to_keep" kwargs if you want.
- [BREAKING] rename with filename starting with lowercase all the files in the "Reward" module. This is 
  both consistent with python practice but allows also to make the difference between the file in the 
  module and the class imported. This should have little to no impact on all codes but to "upgrade"
  instead of `from grid2op.Reward.BaseReward import BaseReward` just do 
  `from grid2op.Reward import BaseReward`.
- [FIXED] an error when an environment with alarm was created before an environment 
  without alert. This lead to a crash when creating the second environment. This is now fixed.
- [FIXED] an issue with non renewable generators in `GymActionSpace` (some curtailment was made
  at 100% of their capacity instead of "no curtailment")
- [FIXED] a bug in computing the datatype of `BoxGymActSpace` and `BoxGymObsSpace` leading to
  using "bool" as dtype when it should be int.
- [FIXED] the behaviour of `BoxGymActSpace` when `subtract` / `divide` were provided (the dtype was 
  not propagated correctly)
- [ADDED] support for the "alert" feature (see main doc page) with new observation attributes
  (`obs.active_alert`, `obs.time_since_last_alert`, `obs.alert_duration`, `obs.total_number_of_alert,` 
  `obs.time_since_last_attack`, `obs.was_alert_used_after_attack` and `obs.attack_under_alert`) 
  a new type of action: `act.raise_alert` and a new reward class `AlertReward` (among others)
- [ADDED] the environment "l2rpn_idf_2023" (accessible via `grid2op.make("l2rpn_idf_2023", test=True)`)
- [ADDED] the `RecoPowerlinePerArea` that is able to reconnect multiple lines in different area in
  the same action
- [ADDED] the kwargs "with_numba" in `PandaPowerBackend` to offer more control on whether or not you want
  to use numba (default behaviour did not change: "if numba is availble, use it" but now you can disable it 
  if numba is available but you don't want it)
- [ADDED] the method `act.decompose_as_unary_actions(...)` to automatically
  decompose a "complex" action on its unary counterpart. 
- [ADDED] the env attribute `env._reward_to_obs` that allows to pass information to the observation directly
  from the reward (this can only be used by regular environment and not by `obs.simulate` nor by `ForecastEnv`)
- [ADDED] the whole "alert" concept in grid2op with a grid2op environment supporting it (`l2rpn_idf_2023`)
- [ADDED] the `gym_env.action_space.get_index(attr_nm)` for `BoxGymActSpace` that allows to retrieve which index
  of the action represents which attribute.
- [ADDED] the argument `quiet_warnings` in the handlers to prevent the issue of too many warnings when using 
  `ForecastHandler`
- [IMPROVED] the method `act.as_serializable_dict()` to work better when exporting / importing actions on different 
  grids (the output dictionary for `set_bus` and `change_bus` now split the keys between all elements types 
  instead of relying on the "topo_vect" order (which might vary))
- [IMPROVED] consistency between how to perform action on storage units between "raw" grid2op, 
  `GymActionSpace`, `BoxGymActSpace`, `DiscreteActSpace` and `MultiDiscreteActSpace` (
    used to be a mix of `set_storage` and `storage_power` now it's consistent and is `set_storage` everywhere)
- [IMPROVED] error message when the "stat.clear_all()" function has been called on a statistic and this same
  statistic is reused.
- [IMPROVED] possibility to set "other_rewards" in the config file of the env

[1.9.0] - 2023-06-06
--------------------
- [BREAKING] (because prone to bug): force the environment name in the `grid2op.make` function.
- [BREAKING] because bugged... The default behaviour for `env.render()` is now "rgb_array". The mode
  "human" has been removed because it needs some fixes. This should not impact lots of code.
- [BREAKING] the "maintenance_forecast" file is deprecated and is no longer used (this should not
  not impact anything)
- [BREAKING] the attribute "connected" as been removed in the edges of the observation converted as
  as a networkx graph. It is replaced by a "nb_connected" attribute. More information on the doc.
- [BREAKING] the function "obs.as_networkx" will be renamed "`obs.get_energy_graph`" and the 
  description has been adapted.
- [BREAKING] In `PandaPowerBackend` the kwargs argument "ligthsim2grid" was misspelled and is now properly
  renamed `lightsim2grid`
- [BREAKING] you can no longer use the `env.reactivate_forecast()` in the middle of an episode.
- [BREAKING] the method `runner.run_one_episode()` (that should not use !) now 
  returns also the total number of steps of the environment.
- [FIXED] a bug in `PandapowerBackend` when running in dc mode (voltages were not read correctly
  from the generators)
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/389 which was caused by 2 independant things: 

  1) the `PandapowerBackend` did not compute the `theta` correctly on powerline especially if
     they are connected to a disconnected bus (in this case I chose to put `theta=0`) 
  2) the `obs.get_energy_graph` (previously `obs.as_networkx()`) method did not check, 
     when updating nodes attributes if powerlines 
     were connected or not, which was wrong in some cases 

- [FIXED] the `N1Reward` that was broken
- [FIXED] the `act._check_for_ambiguity`: a case where missing (when you used topology to disconnect a powerline, 
  but also set_bus to connect it)
- [FIXED] a bug when the storage unit names where not set in the backend and needed to be set
  automatically (wrong names were used)
- [FIXED] a bug in `PandaPowerBackend` when using `BackendConverter` and one the backend do not support shunts.
- [FIXED] 2 issues related to gym env: https://github.com/rte-france/Grid2Op/issues/407 and 
  https://github.com/rte-france/Grid2Op/issues/418
- [FIXED] some bus in the `obs.get_energy_graph` (previously `obs.as_networkx()`) for the cooldowns of substation
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/396
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/403
- [FIXED] a bug in `PandaPowerBackend` when it was copied (the kwargs used to build it were not propagated)
- [FIXED] a bug in the `Runner` when the time series class used is not `MultiFolder` (*eg* `GridStateFromFile`): we could 
  not run twice the same environment. 
- [FIXED] a bug n the `GridStateFromFile`, `GridStateFromFileWithForecasts` and 
  `GridStateFromFileWithForecastsWithoutMaintenance` classes that caused the maintenance file to be 
  ignored when "chunk_size" was set.
- [FIXED] a bug when shunts were alone in `backend.check_kirchoff()`
- [FIXED] an issue with "max_iter" in the runner when `MultifolderWithCache`
  (see issue https://github.com/rte-france/Grid2Op/issues/447)
- [FIXED] a bug in `MultifolderWithCache` when seeding was applied
- [ADDED] the function `obs.get_forecast_env()` that is able to generate a grid2op environment from the
  forecasts data in the observation. This is especially useful in model based RL.
- [ADDED] an example on how to write a backend.
- [ADDED] some convenient function of `gridobject` class to convert back and forth "local bus id" (1 or 2) to
  "global bus id" (0, 1, 2, ... 2*n_sub) [see `gridobject.global_bus_to_local` or `gridobject.local_bus_to_global`]
- [ADDED] a step by step (very detailed) example on how to build a Backend from an existing grid "solver".
- [ADDED] some test when the shunt bus are modified.
- [ADDED] a function to get the "elements graph" from the grid2op observation (represented as a networkx graph)
  as well as its description on the documentation.
- [ADDED] a method to retrieve the "elements graph" (see doc) fom an observation `obs.get_elements_graph()`
- [ADDED] a whole new way to deal with input time series data (see the module `grid2op.Chronics.handlers` 
  for more information)
- [ADDED] possibility to change the parameters used for the `obs.simulate(...)`
  directly from the grid2op action, see `obs.change_forecast_parameters()`
- [ADDED] possibility to retrieve a "forecast environment" with custom forecasts, see 
  `obs.get_env_from_external_forecasts(...)`
- [ADDED] now requires "importlib-metadata" package at install
- [ADDED] adding the `TimedOutEnvironment` that takes "do nothing" actions when the agent
  takes too much time to compute. This involves quite some changes in the runner too.
- [ADDED] Runner is now able to store if an action is legal or ambiguous
- [ADDED] experimental support to count the number of "high resolution simulator" (`obs.simulate`, 
  `obs.get_simulator` and `obs.get_forecast_env`) in the environment (see 
  https://github.com/rte-france/Grid2Op/issues/417). It might not work properly in distributed settings
  (if the agents uses parrallel processing or if MultiProcessEnv is used), in MultiMixEnv, etc.
- [ADDED] it now possible to check the some rules based on the definition of
  areas on the grid.
- [IMPROVED] possibility to "chain" the call to simulate when multiple forecast
- [IMPROVED] possibility to "chain" the call to simulate when multiple forecasts
  horizon are available.
- [IMPROVED] the `GridStateFromFileWithForecasts` is now able to read forecast from multiple steps
  ahead (provided that it knows the horizons in its constructor)
- [IMPROVED] documentation of the gym `DiscreteActSpace`: it is now explicit that the "do nothing" action
  is by default encoded by `0`
- [IMPROVED] documentation of `BaseObservation` and its attributes
- [IMPROVED] `PandapowerBackend` can now be loaded even if the underlying grid does not converge in `AC` (but
  it should still converge in `DC`) see https://github.com/rte-france/Grid2Op/issues/391
- [IMPROVED] `obs.get_energy_graph` (previously `obs.as_networkx()`) method:
  almost all powerlines attributes can now be read from the 
  resulting graph object.
- [IMPROVED] possibility to set `data_feeding_kwargs` from the config file directly.
- [IMPROVED] so "FutureWarnings" are silenced (depending on pandas and pandapower version)
- [IMPROVED] error messages when "env.reset()" has not been called and some functions are not available.
- [IMPROVED] `act.remove_line_status_from_topo` can now be used without an observation and will "remove"
  all the impact on line status from the topology if it causes "AmbiguousAction" (this includes removing
  `set_bus` to 1 or 2 with `set_line_status` is -1 or to remove `set_bus` to -1 when `set_line_status` is 1
  or to remove `change_bus` when `set_line_status` is -1)
- [IMPROVED] possibility, for `BackendConverter` to converter between backends where one does support 
  storage units (the one making powerflow) and the other one don't (the one the user will see).
- [IMPROVED] in `BackendConverter` names of the "source backend" can be used to match the time series data
  when the "use_target_backend_name=True" (new kwargs)
- [IMPROVED] environment do not crash when it fails to load redispatching data. It issues a warning and continue as if
  the description file was not present.
- [IMPROVED] `BackendConverter` is now able to automatically map between different backend with different naming convention 
  under some hypothesis. CAREFUL: the generated mapping might not be the one you "have in mind" ! As for everything automatic,
  it's good because it's fast. It's terrible when you think it does something but in fact it does something else.
- [IMPROVED] the `obs.get_energy_graph` (previously `obs.as_networkx()`) method with added attributes for edges (origin and extremity substation, as well as origin and
  extremity buses)
- [IMPROVED] the doc of the `obs.get_energy_graph` (previously `obs.as_networkx()`)
- [IMPROVED] it is now possible to use a different backend, a different grid or different kwargs between the
  env backend and the obs backend.
- [IMPROVED] the environment now called the "chronics_handler.forecast" function at most once per step.
- [IMPROVED] make it easier to create an environment without `MultiFolder` or `MultifolderWithCache`
- [IMPROVED] add the possibility to forward kwargs to chronix2grid function when calling `env.generate_data`
- [IMPROVED] when calling `env.generate_data` an extra file (json) will be read to set default values 
  passed to `chronix2grid.add_data`
- [IMPROVED] it is no more reasonably possible to misuse the `MultifolderWithCache` (for example by
  forgetting to `reset()` the cache): an error will be raised in case the proper function has not been called.
- [IMPROVED] possibility to pass game rules by instance of object and not by class.
- [IMPROVED] it should be faster to use the "Simulator" (an useless powerflow was run)

[1.8.1] - 2023-01-11
---------------------
- [FIXED] a deprecation with numpy>= 1.24 (**eg** np.bool and np.str)
- [ADDED] the baseAgent class now has two new template methods `save_state` and `load_state` to save and
  load the agent's state during Grid2op simulations. Examples can be found in L2RPN baselines (PandapowerOPFAgent and curriculumagent).
- [IMPROVED] error message in pandapower backend when the grid do not converge due to disconnected
  generators or loads.

[1.8.0] - 2022-12-12
---------------------
- [BREAKING] now requires numpy >= 1.20 to work (otherwise there are 
  issues with newer versions of pandas).
- [BREAKING] issue https://github.com/rte-france/Grid2Op/issues/379 requires
  different behaviour depending on installed gym package.
- [BREAKING] cooldowns are not consistent between `env.step` and `obs.simulate`. 
  If `obs.time_before_cooldown_line[l_id] > 0` it will be illegal, at the next call to `env.step` 
  (and `obs.simulate`) to modify the status of this powerline `l_id`. Same for 
  `obs.time_before_cooldown_sub[s_id] > 0` if trying to modify topology of
  substation `s_id`. This also impacts the maintenances and hazards.
  This is also linked to github issue https://github.com/rte-france/Grid2Op/issues/148
- [FIXED] a bug when using a `Runner` with an environment that has 
  been copied (see https://github.com/rte-france/Grid2Op/issues/361)
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/358
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/363
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/364
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/365 and 
  https://github.com/rte-france/Grid2Op/issues/376 . Now the function(s)
  `gridobj.process_shunt_data` and `gridobj.process_grid2op_shunt_data` are called
  `gridobj.process_shunt_static_data`
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/367
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/369
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/374
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/377 by adding a special
  method `backend.update_thermal_limit_from_vect`
- [ADDED] the "`packaging`" python package is now required to install grid2op. 
  It allows to support different `gym` versions that changes behavior regarding
  numpy pseudo random generator.
- [ADDED] the function `act.remove_line_status_from_topo` to ignore the line status modification
  that would be induced by "set_bus" or "change_bus" when some cooldown applies on the powerline.
- [IMPROVED] clarify documentation of gym compat module (see 
  https://github.com/rte-france/Grid2Op/issues/372 and 
  https://github.com/rte-france/Grid2Op/issues/373) as well as the doc
  for MultifolderWithCache (see https://github.com/rte-france/Grid2Op/issues/370)

[1.7.2] - 2022-07-05
--------------------
- [FIXED] seeding issue https://github.com/rte-france/Grid2Op/issues/331
- [FIXED] clarify doc about fixed size matrices / graphs https://github.com/rte-france/Grid2Op/issues/330
- [FIXED] improved the behaviour of `obs._get_bus_id` and `obs._aux_fun_get_bus` : when some objects were on busbar 2
  they had a "wrong" bus id (it was lagged by 1) meaning an empty "bus" was introduced.
- [FIXED] an issue with `obs.state_of(...)` when inspecting storage units 
  (see https://github.com/rte-france/Grid2Op/issues/340)
- [FIXED] an issue with `act0 + act1` when curtailment was applied 
  (see https://github.com/rte-france/Grid2Op/issues/340)
- [FIXED] a slight "bug" in the formula to compute the redispatching cost for L2RPN 2022 competition.
- [IMPROVED] possibility to pass the env variable `_GRID2OP_FORCE_TEST` to force the flag
  of "test=True" when creating an environment. This is especially useful when testing to prevent
  downloading of data.
- [IMPROVED] support of "kwargs" backend arguments in `MultiMixEnv` see first
  item of version 1.7.1 below

[1.7.1] - 2022-06-03
-----------------------
- [BREAKING] The possibility to propagate keyword arguments between the environment
  and the runner implied adding some arguments in the constructor of 
  `PandapowerBackend`. So if you made a class that inherit from it, you should
  add these arguments in the constructor (otherwise you will not be able to use
  the runner) [This should not impact lot of codes, if any]
- [FIXED] a documentation issue https://github.com/rte-france/Grid2Op/issues/281
- [FIXED] a bug preventing to use the `FromChronix2grid` chronics class when 
  there is an opponent on the grid.
- [FIXED] a documentation issue https://github.com/rte-france/Grid2Op/issues/319
  on notebook 11
- [FIXED] some issues when the backend does not support shunts data (caused during the
  computation of the size of the observation) Tests are now performed in
  `grid2op/tests/test_educpp_backend.py`
- [FIXED] a bug when downloading an environment when the archive name is not the 
  same as the environment names (attempt to delete a non existing folder). This 
  is the case for `l2rpn_wcci_2022` env. For this env, your are forced to use
  grid2op version >= 1.7.1
- [FIXED] an issue when converting a "done" action as a graph, see
  https://github.com/rte-france/Grid2Op/issues/327
- [ADDED] score function for the L2RPN WCCI 2022 competition
- [IMPROVED] adding the compatibility with logger in the reward functions.
- [IMPROVED] when there is a game over caused by redispatching, the observation is
  not updated, as it is the case for other type of game over (improved consistency)
- [IMPROVED] it is now possible to make an environment with a backend that
  cannot be copied.
- [IMPROVED] the arguments used to create a backend can be (if used properly)
  re used (without copy !) when making a `Runner` from an environment for example.
- [IMPROVED] description and definition of `obs.curtailment_limit_effective` are now
  consistent (issue https://github.com/rte-france/Grid2Op/issues/321)

[1.7.0] - 2022-04-29
---------------------
- [BREAKING] the `L2RPNSandBoxScore`, `RedispReward` and `EconomicReward` now properly computes the cost of the grid 
  (there was an error between the conversion from MWh - cost is given in $ / MWh - and MW). 
  This impacts also `ScoreICAPS2021` and `ScoreL2RPN2020`.
- [BREAKING] in the "gym_compat" module the curtailment action type has 
  for dimension the number of dispatchable generators (as opposed to all generators
  before) this was mandatory to fix issue https://github.com/rte-france/Grid2Op/issues/282
- [BREAKING] the size of the continuous action space for the redispatching in
  case of gym compatibility has also been adjusted to be consistent with curtailment.
  Before it has the size of `env.n_gen` now `np.sum(env.gen_redispatchable)`.
- [BREAKING] move the `_ObsEnv` module to `Environment` (was before in `Observation`).
- [BREAKING] adding the `curtailment_limit_effective` in the observation converted to gym. This changes
  the sizes of the gym observation.
- [FIXED] a bug preventing to use `backend.update_from_obs` when there are shunts on the grid for `PandapowerBackend`
- [FIXED] a bug in the gym action space: see issue https://github.com/rte-france/Grid2Op/issues/281
- [FIXED] a bug in the gym box action space: see issue https://github.com/rte-france/Grid2Op/issues/283
- [FIXED] a bug when using `MultifolderWithCache` and `Runner` (see issue https://github.com/rte-france/Grid2Op/issues/285)
- [FIXED] a bug in the `env.train_val_split_random` where sometimes some wrong chronics
  name were sampled.
- [FIXED] the `max` value of the observation space is now 1.3 * pmax to account for the slack bus (it was
  1.01 of pmax before and was not sufficient in some cases)
- [FIXED] a proper exception is added to the "except" kwargs of the "info" return argument of `env.step(...)`
  (previously it was only a string) when redispatching was illegal.
- [FIXED] a bug in `env.train_val_split_random` when some non chronics files where present in the
  "chronics" folder of the environment.
- [FIXED] an error in the redispatching: in some cases, the environment detected that the redispatching was infeasible when it
  was not and in some others it did not detect when it while it was infeasible. This was mainly the case
  when curtailment and storage units were heavily modified.
- [FIXED] now possible to create an environment with the `FromNPY` chronixcs even if the "chronics" folder is absent. 
- [FIXED] a bug preventing to converte observation as networkx graph with oldest version of numpy and newest version of scipy.
- [FIXED] a bug when using `max_iter` and `Runner` in case of max_iter being larger than the number of steps in the
  environment and `nb_episode` >= 2.
- [FIXED] a bug in the hashing of environment in case of storage units (the characteristics of the storage units
  were not taken into account in the hash).
- [FIXED] a bug in the `obs.as_dict()` method.
- [FIXED] a bug in when using the "env.generate_classe()" https://github.com/rte-france/Grid2Op/issues/310
- [FIXED] another bug in when using the "env.generate_classe()" on windows https://github.com/rte-france/Grid2Op/issues/311
- [ADDED] a function `normalize_attr` allowing to easily scale some data for the
  `BoxGymObsSpace` and `BoxGymActSpace`
- [ADDED] support for distributed slack in pandapower (if supported)
- [ADDED] an attribute `self.infos` for the BaseEnv that contains the "info" return value of `env.step(...)`
- [ADDED] the possibility to shuffle the chronics of a `GymEnv` (the default behavior is now to shuffle them)
- [ADDED] two attribtues for the observation: `obs.gen_margin_up` and `obs.gen_margin_down`
- [ADDED] support for hashing chronix2grid related components.
- [ADDED] possibility to change the type of the opponent space type from the `make(...)` command
- [ADDED] a method to "limit the curtailment / storage" action depending on the availability of controllable generators 
  (see `act.limit_curtail_storage(...)`)
- [ADDED] a class to generate data "on the fly" using chronix2grid (for now really slow and only available for 
  a single environment)
- [ADDED] a first version (for testing only) for the `l2rpn_wcci_2022` environment.
- [ADDED] a method to compute the "simple" line reconnection actions (adding 2 actions per lines instead of 5)
  in the action space (see `act_space.get_all_unitary_line_set_simple()`)
- [IMPROVED] better difference between `env_path` and `grid_path` in environments.
- [IMPROVED] addition of a flag to control whether pandapower can use lightsim2grid (to solve the powerflows) or not
- [IMPROVED] clean the warnings issued by pandas when used with pandapower
- [IMPROVED] doc of observation module (some attributes were missing)
- [IMPROVED] officially drop python 3.6 supports (which could not benefit from all the features)
- [IMPROVED] add support for setting the maximum number of iteration in the `PandaPowerBackend`
- [IMPROVED] when the curtailment / storage is too "strong" at a given step, the environment will now allow 
  every controllable turned-on generators to mitigate it. This should increase the possibility to act on the
  curtailment and storage units without "breaking" the environment. 
- [IMPROVED] have dedicated type of actions / observation for L2RPN competition environments, 
  defined in the "conf.py" file (to make possible the use of different
  grid2op version transparently)
- [IMPROVED] on some cases, the routine used to compute the redispatching would lead to a "redispatch" that would
  change even if you don't apply any, for no obvious reasons. This has been adressed, though it's not perfect.
- [IMPROVED] finer resolution when measuring exectution times

[1.6.5] - 2022-01-19
---------------------
- [BREAKING] the function "env.reset()" now reset the underlying pseudo random number generators
  of all the environment subclasses (eg. observation space, action space, etc.) This change has been made to
  ensure reproducibility between episodes: if `env.seed(...)` is called once, then regardless of what happens
  (basically the number of "env.step()" between calls to "env.reset()")
  the "env.reset()" will be generated with the same prng (drawn from the environment)
  This effect the opponent and the chronics (when maintenance are generated "on the fly").
- [BREAKING] the name of the python files for the "Chronics" module are now lowercase (complient with PEP). If you
  did things like `from grid2op.Chronics.ChangeNothing import ChangeNothing` you need to change it like
  `from grid2op.Chronics.changeNothing import ChangeNothing` or even better, and this is the preferred way to include
  them: `from grid2op.Chronics import ChangeNothing`. It should not affect lots of code (more refactoring of the kind
  are to be expected in following versions).
- [BREAKING] same as above for the "Observation" module. It should not affect lots of code (more refactoring of the kind
  are to be expected in following versions).
- [FIXED] a bug for the EpisodeData that did not save the first observation when 
  "add_detailed_output" was set to ``True`` and the data were not saved on disk.
- [FIXED] an issue when copying the environment with the opponent (see issue https://github.com/rte-france/Grid2Op/issues/274)
- [FIXED] a bug leading to the wrong "backend.get_action_to_set()" when there were storage units on the grid. 
- [FIXED] a bug in the "BackendConverter" when there are storage  on the grid
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/265
- [FIXED] issue https://github.com/rte-france/Grid2Op/issues/261
- [ADDED] possibility to "env.set_id" by giving only the folder of the chronics and not the whole path.
- [ADDED] function "env.chronics_handler.available_chronics()" to return the list of available chronics
  for a given environment
- [ADDED] possibility, through the `Parameters` class, to limit the number of possible calls to `obs.simulate(...)` 
  see `param.MAX_SIMULATE_PER_STEP` and `param.MAX_SIMULATE_PER_EPISODE` (see issue https://github.com/rte-france/Grid2Op/issues/273)
- [ADDED] a class to generate a "Chronics" readable by grid2op from numpy arrays (see https://github.com/rte-france/Grid2Op/issues/271)
- [ADDED] an attribute `delta_time` in the observation that tells the time (in minutes) between two consecutive steps.
- [ADDED] a method of the action space to show a list of actions to get back to the original topology 
  (see https://github.com/rte-france/Grid2Op/issues/275)
  `env.action_space.get_back_to_ref_state(obs)`
- [ADDED] a method of the action to store it in a grid2op independant fashion (using json and dictionaries), 
  see `act.as_serializable_dict()`
- [ADDED] possibility to generate a gym `DiscreteActSpace` from a given list of actions (see 
  https://github.com/rte-france/Grid2Op/issues/277)
- [ADDED] a class that output a noisy observation to the agent (see `NoisyObservation`): the agent sees
  the real values of the environment with some noise, this could used to model inacurate
  sensors.
- [IMPROVED] observation now raises `Grid2OpException` instead of `RuntimeError`
- [IMRPOVED] docs (and notebooks) for the "split_train_val" https://github.com/rte-france/Grid2Op/issues/269
- [IMRPOVED] the "`env.split_train_val(...)`" function to also generate a test dataset see 
  https://github.com/rte-france/Grid2Op/issues/276
  
[1.6.4] - 2021-11-08
---------------------
- [BREAKING] the name of the python files for the "agent" module are now lowercase (complient with PEP). If you
  did things like `from grid2op.Agent.BaseAgent import BaseAgent` you need to change it like
  `from grid2op.Agent.baseAgent import BaseAgent` or even better, and this is the preferred way to include
  them: `from grid2op.Agent import BaseAgent`. It should not affect lots of code.
- [FIXED] a bug where the shunt had a voltage when disconnected using pandapower backend
- [FIXED] a bug preventing to print the action space if some "part" of it had no size (empty action space)
- [FIXED] a bug preventing to copy an action properly (especially for the alarm)
- [FIXED] a bug that did not "close" the backend of the observation space when the environment was `closed`. This 
  might be related to `Issue#255 <https://github.com/rte-france/Grid2Op/issues/255>`_
- [ADDED] serialization of `current_iter` and `max_iter` in the observation.
- [ADDED] the possibility to use the runner only on certain episode id
  (see `runner.run(..., episode_id=[xxx, yyy, ...])`)
- [ADDED] a function that returns if an action has any change to modify the grid see `act.can_affect_something()`
- [ADDED] a ttype of agent that performs predefined actions from a given list
- [ADDED] basic support for logging in environment and runner (more coming soon)
- [ADDED] possibility to make an environment with an implementation of a reward, instead of relying on a reward class.
- [ADDED] a possible implementation of a N-1 reward
- [IMPROVED] right time stamp is now set in the observation after the game over.
- [IMPROVED] correct current number of steps when the observation is set to a game over state.
- [IMPROVED] documentation to clearly state that the action_class should not be modified.
- [IMPROVED] possibility to tell which chronics to use with the result of `env.chronics_handler.get_id()` (this is also
  compatible in the runner)
- [IMPROVED] it is no more possible to call "env.reset()" or "env.step()" after an environment has been closed: a clean error
  is raised in this case.

[1.6.3] - 2021-08-21
--------------------
- [FIXED] a bug that allowed to use wrongly the function `backend.get_action_to_set()` even when the backend
  has diverged (which should not be possible)
- [FIXED] a bug leading to non correct consideration of the status of powerlines right after the activation
  of some protections (see `Issue#245 <https://github.com/rte-france/Grid2Op/issues/245>`_ )
- [IMPROVED] the PandaPowerBackend is now able to load a grid with a distributed slack bus. When loaded though, the
  said grid will be converted to one with a single slack bus (the first slack among the distributed)
- [IMPROVED] massive speed-ups when copying environment or using `obs.simulate` (sometimes higher than 30x speed up)
- [IMPROVED] **experimental** compatibility with different frameworks thanks to the possibility to serialize, as text
  files the class created "on the fly" (should solve most of the "pickle" error). See `env.generate_classes()`
  for an example usage. Every feedback is appreciated.

[1.6.2] (hotfix) - 2021-08-18
-----------------------------
- [FIXED] an issue when using `obs.simulate` with `_AlarmScore` (major bug)
- [FIXED] now properly initialized the "complete_action_class" of the backend (minor bug)

[1.6.2] - 2021-07-27
---------------------
- [ADDED] the complete support for pickling grid2op classes. This is a major feature that allows to use grid2op
  way more easily with multiprocessing and to ensure compatibility with more recent version of some RL package
  (*eg* ray / rllib). Note that full compatibility with "multiprocessing" and "pickle" is not completely done yet.

[1.6.1] - 2021-07-27
---------------------
- [FIXED] a bug in the "env.get_path_env()" in case `env` was a multimix (it returned the path of the current mix
  instead of the path of the multimix environment)
- [FIXED] a bug in the `backend.get_action_to_set()` and `backend.update_from_obs()` in case of disconnected shunt
  with backend that supported shunts (values for `p` and `q` were set even if the shunt was disconnected, which
  could lead to undefined behaviour)
- [IMPROVED] now grid2op is able to check if an environment needs to be updated when calling `grid2op.update_env()`
  thanks to the use of registered hash values.
- [IMPROVED] now grid2op will check if an update is available when an environment is being downloaded for the
  first time.

[1.6.0] (hotfix) - 2021-06-23
------------------------------
- [FIXED] issue `Issue#235 <https://github.com/rte-france/Grid2Op/issues/235>`_ issue when using the "simulate"
  feature in case of divergence of powerflow.

[1.6.0] - 2021-06-22
--------------------
- [BREAKING] (but transparent for everyone): the `disc_lines` attribute is now part of the environment, and is also
  containing integer (representing the "order" on which the lines are disconnected due to protections) rather
  than just boolean.
- [BREAKING] now the observation stores the information related to shunts by default. This means old logs computed with
  the runner might not work with this new version.
- [BREAKING] the "Runner.py" file has been renamed, following pep convention "runner.py". You should rename your
  import `from grid2op.Runner.Runner import Runner` to `from grid2op.Runner.runner import Runner`
  (**NB** we higly recommend importing the `Runner` like `from grid2op.Runner import Runner` though !)
- [FIXED]: the L2RPN_2020 score has been updated to reflect the score used during these competitions (there was an
  error between `DoNothingAgent` and `RecoPowerlineAgent`)
  [see `Issue#228 <https://github.com/rte-france/Grid2Op/issues/228>`_ ]
- [FIXED]: some bugs in the `action_space.get_all_unitary_redispatch` and `action_space.get_all_unitary_curtail`
- [FIXED]: some bugs in the `GreedyAgent` and `TopologyGreedy`
- [FIXED]: `Issue#220 <https://github.com/rte-france/Grid2Op/issues/220>`_ `flow_bus_matrix` did not took into
  account disconnected powerlines, leading to impossibility to compute this matrix in some cases.
- [FIXED]: `Issue#223 <https://github.com/rte-france/Grid2Op/issues/223>`_ : now able to plot a grid even
  if there is nothing controllable in grid2op present in it.
- [FIXED]: an issue where the parameters would not be completely saved when saved in json format (alarm feature was
  absent) (related to `Issue#224 <https://github.com/rte-france/Grid2Op/issues/224>`_ )
- [FIXED]: an error caused by the observation non being copied when a game over occurred that caused some issue in
  some cases (related to `Issue#226 <https://github.com/rte-france/Grid2Op/issues/226>`_ )
- [FIXED]: a bug in the opponent space where the "`previous_fail`" kwargs was not updated properly and send wrongly
  to the opponent
- [FIXED]: a bug in the geometric opponent when it did attack that failed.
- [FIXED]: `Issue#229 <https://github.com/rte-france/Grid2Op/issues/229>`_ typo in the  `AlarmReward` class when reset.
- [ADDED] support for the "alarm operator" / "attention budget" feature
- [ADDED] retrieval of the `max_step` (ie the maximum number of step that can be performed for the current episode)
  in the observation
- [ADDED] some handy argument in the `action_space.get_all_unitary_redispatch` and
  `action_space.get_all_unitary_curtail` (see doc)
- [ADDED] as utils function to compute the score used for the ICAPS 2021 competition (see
  `from grid2op.utils import ScoreICAPS2021` and the associate documentation for more information)
- [ADDED] a first version of the "l2rpn_icaps_2021" environment (accessible with
  `grid2op.make("l2rpn_icaps_2021", test=True)`)
- [IMPROVED] prevent the use of the same instance of a backend in different environments
- [IMPROVED] `Issue#217 <https://github.com/rte-france/Grid2Op/issues/217>`_ : no more errors when trying to
  load a grid with unsupported elements (eg. 3w trafos or static generators) by PandaPowerBackend
- [IMPROVED] `Issue#215 <https://github.com/rte-france/Grid2Op/issues/215>`_ : warnings are issued when elements
  present in pandapower grid will not be modified grid2op side.
- [IMPROVED] `Issue#214 <https://github.com/rte-france/Grid2Op/issues/214>`_ : adding the shunt information
  in the observation documentation.
- [IMPROVED] documentation to use the `env.change_paramters` function.

[1.5.2] - 2021-05-10
-----------------------
- [BREAKING]: allow the opponent to chose the duration of its attack. This breaks the previous "Opponent.attack(...)"
  signature by adding an object in the return value. All code provided with grid2op are compatible with this
  new change. (for previously coded opponent, the only thing you have to do to make it compliant with
  the new interface is, in the `opponent.attack(...)` function return `whatever_you_returned_before, None` instead
  of simply `whatever_you_returned_before`)
- [FIXED]: `Issue#196 <https://github.com/rte-france/Grid2Op/issues/196>`_ an issue related to the
  low / high of the observation if using the gym_compat module. Some more protections
  are enforced now.
- [FIXED]: `Issue#196 <https://github.com/rte-france/Grid2Op/issues/196>`_ an issue related the scaling when negative
  numbers are used (in these cases low / max would be mixed up)
- [FIXED]: an issue with the `IncreasingFlatReward` reward types
- [FIXED]: a bug due to the conversion of int to float in the range of the `BoxActionSpace` for the `gym_compat` module
- [FIXED]: a bug in the `BoxGymActSpace`, `BoxGymObsSpace`, `MultiDiscreteActSpace` and `DiscreteActSpace`
  where the order of the attribute for the conversion
  was encoded in a set. We enforced a sorted list now. We did not manage to find a bug caused by this issue, but
  it is definitely possible. This has been fixed now.
- [FIXED]: a bug where, when an observation was set to a "game over" state, some of its attributes were below the
  maximum values allowed in the `BoxGymObsSpace`
- [ADDED]: a reward `EpisodeDurationReward` that is always 0 unless at the end of an episode where it returns a float
  proportional to the number of step made from the beginning of the environment.
- [ADDED]: in the `Observation` the possibility to retrieve the current number of steps
- [ADDED]: easier function to manipulate the max number of iteration we want to perform directly from the environment
- [ADDED]: function to retrieve the maximum duration of the current episode.
- [ADDED]: a new kind of opponent that is able to attack at "more random" times with "more random" duration.
  See the `GeometricOpponent`.
- [IMPROVED]: on windows at least, grid2op does not work with gym < 0.17.2 Checks are performed in order to make sure
  the installed open ai gym package meets this requirement (see issue
  `Issue#185 <https://github.com/rte-france/Grid2Op/issues/185>`_ )
- [IMPROVED] the seed of openAI gym for composed action space (see issue `https://github.com/openai/gym/issues/2166`):
  in waiting for an official fix, grid2op will use the solution proposed there
  https://github.com/openai/gym/issues/2166#issuecomment-803984619

[1.5.1] - 2021-04-15
-----------------------
- [FIXED]: `Issue#194 <https://github.com/rte-france/Grid2Op/issues/194>`_: (post release): change the name
  of the file `platform.py` that could be mixed with the python "platform" module to `_glop_platform_info.py`
- [FIXED]: `Issue #187 <https://github.com/rte-france/Grid2Op/issues/187>`_: improve the computation and the
  documentation of the `RedispReward`. This has an impact on the `env.reward_range` of all environments using this
  reward, because the old "reward_max" was not correct.
- [FIXED] `Issue #181 <https://github.com/rte-france/Grid2Op/issues/181>`_ : now environment can be created with
  a layout and a warning is issued in this case.
- [FIXED] `Issue #180 <https://github.com/rte-france/Grid2Op/issues/180>`_ : it is now possible to set the thermal
  limit with a dictionary
- [FIXED] a typo that would cause the attack to be discarded in the runner in some cases (cases for now not used)
- [FIXED] an issue linked to the transformation into gym box space for some environments,
  this **might** be linked to `Issue #185 <https://github.com/rte-france/Grid2Op/issues/185>`_
- [ADDED] a feature to retrieve the voltage angle (theta) in the backend (`backend.get_theta`) and in the observation.
- [ADDED] support for multimix in the GymEnv (lack of support spotted thanks to
  `Issue #185 <https://github.com/rte-france/Grid2Op/issues/185>`_ )
- [ADDED] basic documentation of the environment available.
- [ADDED] `Issue #166 <https://github.com/rte-france/Grid2Op/issues/166>`_ : support for simulate in multi environment
  settings.
- [IMPROVED] extra layer of security preventing modification of `observation_space` and `action_space` of environment
- [IMPROVED] better handling of dynamically generated classes
- [IMPROVED] the documentation of the opponent

[1.5.0] - 2021-03-31
-------------------------
- [BREAKING] `backend.check_kirchoff()` method now returns also the discrepancy in the voltage magnitude
  and not only the error in the P and Q injected at each bus.
- [BREAKING] the class method "to_dict" used to serialize the action_space and observation_space has been
  renamed `cls_to_dict` to avoid confusion with the `to_dict` method of action and observation (that stores,
  as dictionary the instance of the action / observation). It is now then possible to serialize the action class
  used and the observation class used as dictionary to (using `action.cls_to_dict`)
- [BREAKING] for backend class implementation: need to upgrade your code to take into account the storage units
  if some are present in the grid even if you don't want to use storage units.
- [BREAKING] the backend `runpf` method now returns a flag indicating if the simulation was successful AND (new)
  the exception in case there are some (it now returns a tuple). This change only affect new Backends.
- [BREAKING] rename the attribute "parameters" of the "observation_space" to `_simulate_parameters` to avoid
  confusion with the `parameters` attributes of the environment.
- [BREAKING] change of behaviour of the `env.parameters` attribute behaviour. It is no more possible to
  modified it with `env.parameters = ...` and the `env.parameters.PARAM_ATTRIBUTE = xxx` will have not effect
  at all. Use `env.change_parameters(new_parameters)` for changing the environment parameters and
  `env.change_forecast_parameters(new_param_for_simulate)` for changing the parameters used for simulate.
  (**NB** in both case you need to perform a "env.reset()" for the new parameters to be used. Any attempt to use
  an environment without a call to 'env.reset()' will lead to undefined behaviour).
- [BREAKING] `env.obs_space.rewardClass` is not private and is called `env.obs_space._reward_func`. To change
  this function, you need to call `env.change_reward(...)`
- [BREAKING] more consistency in the observation attribute names, they are now `gen_p`, `gen_q` and `gen_v`
  instead of `prod_p`, `prod_q` and `prod_v` (old names are still accessible for backward compatibility
  in the observation space) but
  conversion to json / dict will be affected as well as the converters (*eg* for gym compatibility)
- [FIXED] `Issue #164 <https://github.com/rte-france/Grid2Op/issues/164>`_: reward is now properly computed
  at the end of an episode.
- [FIXED] A bug where after running a Runner, the corresponding EpisodeData's CollectionWrapper where not properly updated,
  and did not contain any objects.
- [FIXED] A bug when the opponent should chose an attack with all lines having flow 0, but one being still connected.
- [FIXED] An error in the `obs.flow_bus_matrix` when `active_flow=False` and there were shunts on the
  powergrid.
- [FIXED] `obs.connectivity_matrix` now properly takes into account when two objects are disconnected (before
  it was as if there were connected together)
- [FIXED] some surprising behaviour when using  `obs.simulate` just before or just after a planned
  maintenance operation.
- [FIXED] a minimal bug in the `env.copy` method (the wrong simulated backend was used in the observation at
  right after the copy).
- [FIXED] a bug in the serialization (as vector) of some action classes, namely: `PowerlineSetAction` and
  `PowerlineSetAndDispatchAction` and `PowerlineChangeDispatchAndStorageAction`
- [FIXED] a bug preventing to use the `obs.XXX_matrix()` function twice
- [FIXED] issue `Issue #172 <https://github.com/rte-france/Grid2Op/issues/172>`_: wrong assertion was made preventing
  the use of `env.train_val_split_random()`
- [FIXED] issue `Issue #173 <https://github.com/rte-france/Grid2Op/issues/173>`_: a full nan vector could be
  converted to action or observation without any issue if it had the proper dimension. This was due to a conversion
  to integer from float.
- [FIXED] an issue preventing to load the grid2op.utils submodule when installed not in "develop" mode
- [FIXED] some issue with the multiprocessing of the runner on windows
- [ADDED] more complete documentation for the runner.
- [ADDED] a convenient function to evaluate the impact (especially on topology) of an action on a state
  (`obs + act`)
- [ADDED] a property to retrieve the thermal limits from the observation.
- [ADDED] documentation of the main elements of the grid and their "modeling" in grid2op.
- [ADDED] parameters are now checked and refused if not valid (a RuntimeError is raised)
- [ADDED] support for storage unit in grid2op (analog as a "load" convention positive: power absorbed from the grid,
  negative: power given to the grid having some energy limit and power limit). A new object if added in the substation.
- [ADDED] Support for sparse matrices in `obs.bus_connectivity_matrix`
- [ADDED] In the observation, it is now possible to retrieve the "active flow graph" (ie graph with edges having active
  flows, and nodes the active production / consumption) and "reactive flow graph" (see `flow_bus_matrix`)
- [ADDED] more consistent behaviour when using the action space across the different type of actions.
  Now it should understand much more way to interact with it.
- [ADDED] lots of action properties to manipulate action in a more pythonic way, for example using
  `act.load_set_bus = ...` instead of the previously way more verbose `act.update({"set_bus": {"loads_id": ...}})`
  (this applies for `load`, `gen`, `storage`, `line_or` and `line_ex` and to `set_bus` and `change_bus` and
  also to `storage_p` and `redispatch` so making 12 "properties" in total)
- [ADDED] an option to retrieve in memory the `EpisodeData` of each episode computed when using the runner.
  see `runner.run(..., add_detailed_output=True)`
- [ADDED] the option `as_csr_matrix` in `obs.connectivity_matrix` function
- [ADDED] convenient option to get the topology of a substation from an observation (`obs.sub_topology(sub_id=...)`)
- [ADDED] some basic tests for the environments shipped with grid2op.
- [ADDED] grid2op now ships with the `l2rpn_case14_sandbox` environment
- [ADDED] a function to list environments available for testing / illustration purpose.
- [ADDED] a function of the observation to convert it to a networkx graph (`obs.as_networkx()`)
- [ADDED] support for curtailment feature in grid2op (curtailment on the renewable generator units).
- [ADDED] better backward compatibility when reading data generated with previous grid2op version.
- [IMPROVED] simplify the interface for the gym converter.
- [IMPROVED] simplify the interface for the `env.train_val_split` and `env.train_val_split_random`
- [IMPROVED] print of an action now limits the number of decimal for redispatching and storage units

[1.4.0] - 2020-12-10
----------------------
- [CHANGED] The parameters `FORECAST_DC` is now deprecated. Please use
  `change_forecast_parameters(new_param)` with `new_param.ENV_DC=...` instead.
- [FIXED] and test the method `backend.get_action_to_set`
- [FIXED] an error for the voltage of the shunt in the `PandapowerBackend`
- [FIXED] `PowerLineSet` and `PowerSetAndDispatch` action were not properly converted to vector.
- [ADDED] a method to set the state of a backend given a complete observation.
- [ADDED] a `utils` module to store the data of some environment and be able to compute the scores (as in the neurips
  l2rpn competitions). This module might move at a different place in the future
- [ADDED] a function to "split" an environment into train / validation using `os.symlink`
- [ADDED] the implementation of `+` operator for action (based on previously available `+=`)
- [ADDED] A more detailed documentation on the representation of the topology and how to create a backend
- [ADDED] A easier way to set up the topology in backend (eg. `get_loads_bus`)
- [ADDED] A easier way to set up the backend, with automatic computation of some attributes (eg. `*_to_sub_pos`,
  `sub_info`, `dim_topo`) if needed.
- [ADDED] A function to change the `parameters` used by the environment (or `obs_env`) "on the fly" (has only impact
  AFTER `env.reset` is called) (see `change_parameters` and `change_forecast_parameters`)
- [IMPROVED] `PandaPowerBackend` now should take less time to when `reset`.
- [IMPROVED] some speed up in the grid2op computation

[1.3.1] - 2020-11-04
----------------------
- [FIXED] the environment "educ_case14_redisp"
- [FIXED] notebooks are now working perfectly

[1.3.0] - 2020-11-02
---------------------
- [BREAKING] GymConverter has been moved to `grid2op.gym_compat` module instead of  `grid2op.Converter`
- [FIXED] wrong computation of voltage magnitude at extremity of powerlines when the powerlines were disconnected.
- [FIXED] `Issue #151 <https://github.com/rte-france/Grid2Op/issues/151>`_: modification of observation attributes 3
  could lead to crash
- [FIXED] `Issue #153 <https://github.com/rte-france/Grid2Op/issues/153>`_: negative generator could happen in some
  cases
- [FIXED] an error that lead to wrong normalization of some generator (due to slack bus) when using the
  gymconverter.
- [FIXED] a bug that prevented runner to read back previously stored data (and now a test to check
  backward compatibility down to version 1.0.0)
- [FIXED] small issue that could lead to non reproducibility when shuffling chronics
- [FIXED] a bug in `obs.bus_connectivity_matrix()` when powerlines were disconnected
- [ADDED] a class to deactivate the maintenance and hazards in the chronics from file
  `GridStateFromFileWithForecastsWithoutMaintenance`
- [ADDED] a keyword argument in the matplotlib plot information on the grid
  (`plot_helper.plot_info(..., coloring=...)`)
- [ADDED] a function to change the color palette of powerlines (`plot_helper.assign_line_palette`)
- [ADDED] a function to change the color palette of generators (`plot_helper.assign_gen_palette`)
- [ADDED] Support the attack of the opponent in the `EpisodeData` class
- [ADDED] Now the observations are set to a "game over" state when a game over occurred
  see `BaseObservation.set_game_over`
- [ADDED] a method to plot the redispatching state of the grid `PlotMatplot.plot_current_dispatch`
- [ADDED] the documentation of `Episode` module that was not displayed.
- [IMPROVED] silence the warning issue when calling `MultiEnv.get_seeds`
- [IMPROVED] the tolerance of the redispatching algorithm is now more consistent between the precision of the solver
  used and the time when it's
- [IMPROVED] make faster and more robust the optimization routine used during redispatching
- [IMPROVED] error message when the state fails because of infeasible redispatching

[1.2.3] - 2020-09-25
----------------------
- [ADDED] `l2rpn-baselines` package dependency in the "binder" environment.
- [FIXED] binder integration that was broken momentarily
- [FIXED] an issue in the sampling of redispatching action (ramp up and ramp down were inverted)
- [FIXED] an issue causing errors when using `action_space.change_bus` and `action_space.set_bus`
- [FIXED] an issue in the sampling: redispatching and "change_bus" where always performed at the
  same time
- [FIXED] `Issue #144 <https://github.com/rte-france/Grid2Op/issues/144>`_: typo that could lead to not
  display some error messages in some cases.
- [FIXED] `Issue #146 <https://github.com/rte-france/Grid2Op/issues/146>`_: awkward behaviour that lead to not calling
  the reward function when the episode was over.
- [FIXED] `Issue #147 <https://github.com/rte-france/Grid2Op/issues/147>`_: un consistency between step and simulate
  when cooldowns where applied (rule checking was not using the right method).
- [FIXED] An error preventing the loading of an Ambiguous Action (in case an agent took such action, the `EpisodeData`
  would not load it properly).
- [IMPROVED] overall documentation of `BaseEnv` and `Environment`
- [IMPROVED] rationalize the public and private part of the API for `Environment` and `BaseEnv`.
  Some members have been moved to private attribute (their modification would largely alterate the
  behaviour of grid2op).
- [IMPROVED] internal functions are tagged as "Internal, do not use" in the documentation.
- [IMPROVED] Improved documentation for the `Environment` and `MultiMixEnvironment`.

[1.2.2] - 2020-08-19
---------------------
- [FIXED] `LightSim Issue #10<https://github.com/BDonnot/lightsim2grid/issues/10>`_: tests were
  not covering every usecase

[1.2.1] - 2020-08-18
---------------------
- [ADDED] a function that allows to modify some parameters of the environment (see `grid2op.update_env`)
- [ADDED] a class to convert between two backends
- [FIXED] out dated documentation in some classes
- [FIXED] `Issue #140<https://github.com/rte-france/Grid2Op/issues/140>`_: illegal action were
  not properly computed in some cases, especially in case of divergence of the powerflow. Also now
  the "why" the action is illegal is displayed (instead of a generic "this action is illegal").
- [FIXED] `LightSim Issue #10<https://github.com/BDonnot/lightsim2grid/issues/10>`_:
  copy of whole environments without needing pickle module.
- [UPDATED] a missing class documentation `Chronics.Multifolder` in that case.

[1.2.0] - 2020-08-03
---------------------
- [ADDED] `ActionSpace.sample` method is now implemented
- [ADDED] DeltaRedispatchRandomAgent: that takes redispatching actions of a configurable [-delta;+delta] in MW on random generators.
- [FIXED] `Issue #129<https://github.com/rte-france/Grid2Op/issues/129>`_: game over count for env_actions
- [FIXED] `Issue #127 <https://github.com/rte-france/Grid2Op/issues/127>`_: Removed no longer existing attribute docstring `indisponibility`
- [FIXED] `Issue #133 <https://github.com/rte-france/Grid2Op/issues/133>`_: Missing positional argument `space_prng` in `Action.SerializableActionSpace`
- [FIXED] `Issue #131 <https://github.com/rte-france/Grid2Op/issues/131>`_: Forecast values are accessible without needing to call `obs.simulate` beforehand.
- [FIXED] `Issue #134 <https://github.com/rte-france/Grid2Op/issues/134>`_: Backend iadd actions with lines extremities disconnections (set -1)
- [FIXED] issue `Issue #125 <https://github.com/rte-france/Grid2Op/issues/125>`_
- [FIXED] issue `Issue #126 <https://github.com/rte-france/Grid2Op/issues/126>`_ Loading runner logs no longer checks environment actions ambiguity
- [IMPROVED] issue `Issue #16 <https://github.com/rte-france/Grid2Op/issues/16>`_ improving openai gym integration.
- [IMPROVED] `Issue #134 <https://github.com/rte-france/Grid2Op/issues/134>`_ lead us to review and rationalize the
  behavior of grid2op concerning the powerline status. Now it behave more rationally and has now the following
  behavior: if a powerline origin / extremity bus is "set" to -1 at one end and not modified at the other, it will disconnect this
  powerline, if a powerline origin / extremity  bus is "set" to 1 or 2 at one end and not modified at the other, it will
  reconnect the powerline. If a powerline bus is "set" to -1 at one end and set to 1 or 2 at its other
  end the action is ambiguous.
- [IMPROVED] way to count what is affect by an action (affect the cooldown of substation and powerline
  and the legality of some action). And action disconnect a powerline (using the "set_bus") will be
  considered to affect only
  this powerline (and not on its substations) if and only if the powerline was connected (otherwise it
  affects also on the substation). An action that connects a powerline (using the "set_bus") will affect
  only this powerline (and not its substations) if and only if this powerline was disconnected (
  otherwise it affects the substations but not the powerline). Changing the bus of an extremity of
  a powerline if this powerline is connected has no impact on its status and therefor it considers
  it only affects the corresponding substation.
- [IMPROVED] added documentation and usage example for `CombineReward` and `CombineScaledReward`

[1.1.1] - 2020-07-07
---------------------
- [FIXED] the EpisodeData now properly propagates the end of the episode
- [FIXED] `MultiFolder.split_and_save` function did not use properly the "seed"
- [FIXED] issue `Issue 122 <https://github.com/rte-france/Grid2Op/issues/122>`_
- [FIXED] Loading of multimix environment when they are already present in the data cache.
- [UPDATED] notebook 3 to reflect the change made a long time ago for the ambiguous action
  (when a powerline is reconnected)

[1.1.0] - 2020-07-03
---------------------
- [FIXED] forgot to print the name of the missing environment when error in creating it.
- [FIXED] an issue in `MultiFolder.sample_next_chronics` that did not returns the right index
- [FIXED] an issue that prevented the `EpisodeData` class to load back properly the action of the environment.
  This might have side effect if you used the `obs.from_vect` or `act.from_vect` in non conventional ways.
- [ADDED] some documentation and example for the `MultiProcessEnv`
- [IMPROVED] check that the sub environments are suitable grid2op.Environment.Environment in multiprocess env.
- [FIXED] Minor documentation generation warnings and typos (Parameters, Backend, OpponentSpace, ActionSpace)

[1.0.0] - 2020-06-24
---------------------
- [BREAKING] `MultiEnv` has been renamed `SingleEnvMultiProcess`
- [BREAKING] `MultiEnv` has been abstracted to `BaseMultiProcessEnv` and the backwards compatible interface is now
  `SingleProcessMultiEnv`
- [BREAKING] the `seeds` parameters of the `Runner.run` function has been renamed `env_seeds` and an `agent_seeds`
  parameters is now available for fully reproducible experiments.
- [FIXED] a weird effect on `env.reset` that did not reset the state of the previous observation held
  by the environment. This could have caused some issue in some corner cases.
- [FIXED] `BaseAction.__iadd__` fixed a bug with change actions `+=` operator reported in
  `Issue #116 <https://github.com/rte-france/Grid2Op/issues/116>`_
- [FIXED] `obs.simulate` post-initialized reward behaves like the environment
- [FIXED] `LinesReconnectedReward` fixes reward inverted range
- [FIXED] the `get_all_unitary_topologies_change` now counts only once the "do nothing" action.
- [FIXED] `obs.simulate` could sometime returns "None" when the simulated action lead to a game over. This is no longer
  a problem.
- [FIXED] `grid2op.make` will now raise an error if an invalid argument has been passed to it.
- [FIXED] some arguments were not passed correctly to `env.get_kwargs()` or `env.get_params_for_runner()`
- [ADDED] `Issue #110 <https://github.com/rte-france/Grid2Op/issues/110>`_ Adding an agent that is able to reconnect
  disconnected powerlines that can be reconnected, see `grid2op.Agent.RecoPowerlineAgent`
- [ADDED] a clearer explanation between illegal and ambiguous action.
- [ADDED] `MultiEnvMultiProcess` as a new multi-process class to run different environments in multiples prallel
  processes.
- [ADDED] more control on the environment when using the `grid2op.make` function.
- [ADDED] creation of the MultiMixEnv that allows to have, through a unified interface the possibility to interact
  alternatively with one environment or the other. This is especially useful when considering an agent that should
  interact in multiple environments.
- [ADDED] possibility to use `simulate` on the current observation.
- [ADDED] the overload of "__getattr__" for environment running in parallel
- [ADDED] capability to change the powerlines on which the opponent attack at the environment initialization
- [UPDATED] `Backend.PandaPowerBackend.apply_action` vectorized backend apply action method for speed.
- [UPDATED] `Issue #111 <https://github.com/rte-france/Grid2Op/issues/111>`_ Converter is better documented to be
  more broadly usable.
- [UPDATED] `MultiEnv` has been updated for new use case: Providing different environments configurations on the same
  grid and an arbitrary number of processes for each of these.
- [UPDATED] Behaviour of "change_bus" and "set_bus": it is no more possible to affect the bus of a powerline
  disconnected.
- [UPDATED] More control about the looping strategy of the `ChronicsHandler` that has been refactored, and can now be
  more easily cached (no need to do an expensive reading of the data at each call to `env.reset`)

[0.9.4] - 2020-06-12
---------------------
- [FIXED] `Issue #114 <https://github.com/rte-france/Grid2Op/issues/114>`_ the issue concerning the
  bug for the maintenance.


[0.9.3] - 2020-05-29
---------------------
- [FIXED] `Issue #69 <https://github.com/rte-france/Grid2Op/issues/69>`_ MultEnvironment is now working with windows
  based OS.
- [ADDED] `Issue #108 <https://github.com/rte-france/Grid2Op/issues/108>`_ Seed is now part of the public agent API.
  The notebook has been updated accordingly.
- [ADDED] Some function to disable the `obs.simulate` if wanted. This can lead to around 10~15% performance speed up
  in case `obs.simulate` is not used. See `env.deactivate_forecast` and `env.reactivate_forecast`
  (related to `Issued #98 <https://github.com/rte-france/Grid2Op/issues/98>`_)
- [UPDATED] the first introductory notebook.
- [UPDATED] possibility to reconnect / disconnect powerline giving its name when using `reconnect_powerline` and
  `disconnect_powerline` methods of the action space.
- [UPDATED] `Issue #105 <https://github.com/rte-france/Grid2Op/issues/105>`_ problem solved for notebook 4.
  based OS.
- [UPDATED] overall speed enhancement mostly in the `VoltageControler`, with the adding of the previous capability,
  some updates in the `BackendAction`
  `Issued #98 <https://github.com/rte-france/Grid2Op/issues/98>`_
- [UPDATED] Added `PlotMatplot` constructor arguments to control display of names and IDs of the grid elements
  (gen, load, lines). As suggested in `Issue #106 <https://github.com/rte-france/Grid2Op/issues/106>`_


[0.9.2] - 2020-05-26
---------------------
- [FIXED] `GridObject` loading from file does initialize single values (`bool`, `int`, `float`)
  correctly instead of creating a `np.array` of size one.
- [FIXED] `IdToAct` loading actions from file .npy
- [FIXED] a problem on the grid name import on some version of pandas
- [ADDED] a function that returns the types of the action see `action.get_types()`
- [ADDED] a class to "cache" the data in memory instead of reading it over an over again from disk (see
  `grid2op.chronics.MultifolderWithCache` (related to
  `Issued #98 <https://github.com/rte-france/Grid2Op/issues/98>`_) )
- [ADDED] improve the documentation of the observation class.
- [UPDATED] Reward `LinesReconnectedReward` to take into account maintenances downtimes
- [UPDATED] Adds an option to disable plotting load and generators names when using `PlotMatplot`

[0.9.1] - 2020-05-20
---------------------
- [FIXED] a bug preventing to save gif with episode replay when there has been a game over before starting time step
- [FIXED] the issue of the random seed used in the environment for the runner.

[0.9.0] - 2020-05-19
----------------------
- [BREAKING] `Issue #83 <https://github.com/rte-france/Grid2Op/issues/83>`_: attributes name of the Parameters class
  are now more consistent with the rest of the package. Use `NB_TIMESTEP_OVERFLOW_ALLOWED`
  instead of `NB_TIMESTEP_POWERFLOW_ALLOWED`, `NB_TIMESTEP_COOLDOWN_LINE` instead of `NB_TIMESTEP_LINE_STATUS_REMODIF`
  and `NB_TIMESTEP_COOLDOWN_SUB` instead of `NB_TIMESTEP_TOPOLOGY_REMODIF`
- [BREAKING] `Issue #87 <https://github.com/rte-france/Grid2Op/issues/87>`_: algorithm of the environment that solves
  the redispatching to make sure the environment meet the phyiscal constraints is now cast into an optimization
  routine that uses `scipy.minimize` to be solved. This has a few consequences: more dispatch actions are tolerated,
  computation time can be increased in some cases, when the optimization problem cannot be solved, a game
  over is thrown, `scipy` is now a direct dependency of `grid2op`, code base of `grid2op` is simpler.
- [BREAKING] any attempt to use an un intialized environment (*eg* after a game over but before calling `env.reset`
  will now raise a `Grid2OpException`)
- [FIXED] `Issue #84 <https://github.com/rte-france/Grid2Op/issues/84>`_: it is now possible to load multiple
  environments in the same python script and perform random action on each.
- [FIXED] `Issue #86 <https://github.com/rte-france/Grid2Op/issues/86>`_: the proper symmetries are used to generate
  all the actions that can "change" the buses (`SerializationActionSpace.get_all_unitary_topologies_change`).
- [FIXED] `Issue #88 <https://github.com/rte-france/Grid2Op/issues/88>`_: two flags are now used to tell the environment
  whether or not to activate the possibility to dispatch a turned on generator (`forbid_dispatch_off`) and whether
  or not to ignore the gen_min_uptimes and gen_min_downtime propertiers (`ignore_min_up_down_times`) that
  are initialized from the Parameters of the grid now.
- [FIXED] `Issue #89 <https://github.com/rte-france/Grid2Op/issues/89>`_: pandapower backend should not be compatible
  with changing the bus of the generator representing the slack bus.
- [FIXED] Greedy agents now uses the proper data types `dt_float` for the simulated reward (previously it was platform
  dependant)
- [ADDED] A way to limit `EpisodeReplay` to a specific part of the episode. Two arguments have been added, namely:
  `start_step` and `end_step` that default to the full episode duration.
- [ADDED] more flexibilities in `IdToAct` converter not to generate every action for both set and change for example.
  This class can also serialize and de serialize the list of all actions with the save method (to serialize) and the
  `init_converter` method (to read back the data).
- [ADDED] a feature to have multiple difficulty levels per dataset.
- [ADDED] a converter to transform prediction in connectivity of element into valid grid2op action. See
  `Converter.ConnectivitiyConverter` for more information.
- [ADDED] a better control for the seeding strategy in `Environment` and `MultiEnvironment` to improve the
  reproducibility of the experiments.
- [ADDED] a chronics class that is able to generate maintenance data "on the fly" instead of reading the from a file.
  This class is particularly handy to train agents with different kind of maintenance schedule.

[0.8.2] - 2020-05-13
----------------------
- [FIXED] `Issue #75 <https://github.com/rte-france/Grid2Op/issues/75>`_: PlotGrid displays double powerlines correctly.
- [FIXED] Action `+=` operator (aka. `__iadd__`) doesn't create warnings when manipulating identical arrays
  containing `NaN` values.
- [FIXED] `Issue #70 <https://github.com/rte-france/Grid2Op/issues/70>`_: for powerline disconnected, now the voltage
  is properly set to `0.0`
- [UPDATED] `Issue #40 <https://github.com/rte-france/Grid2Op/issues/40>`_: now it is possible to retrieve the forecast
  of the injections without running an expensive "simulate" thanks to the `obs.get_forecasted_inj` method.
- [UPDATED] `Issue #78 <https://github.com/rte-france/Grid2Op/issues/78>`_: parameters can be put as json in the
  folder of the environment.
- [UPDATED] minor fix for `env.make`
- [UPDATED] Challenge tensorflow dependency to `tensorflow==2.2.0`
- [UPDATED] `make` documentation to reflect API changes of 0.8.0

[0.8.1] - 2020-05-05
----------------------
- [FIXED] `Issue #65 <https://github.com/rte-france/Grid2Op/issues/65>`_: now the length of the Episode Data is properly
  computed
- [FIXED] `Issue #66 <https://github.com/rte-france/Grid2Op/issues/66>`_: runner is now compatible with multiprocessing
  again
- [FIXED] `Issue #67 <https://github.com/rte-france/Grid2Op/issues/67>`_: L2RPNSandBoxReward is now properly computed
- [FIXED] Serialization / de serialization of Parameters as json is now fixed

[0.8.0] - 2020-05-04
----------------------
- [BREAKING] All previously deprecated features have been removed
- [BREAKING] `grid2op.Runner` is now located into a submodule folder
- [BREAKING]  merge of `env.time_before_line_reconnectable` into `env.times_before_line_status_actionable` which
  referred to
  the same idea: impossibility to reconnect a powerilne. **Side effect** observation have a different size now (
  merging of `obs.time_before_line_reconnectable` into `obs.time_before_cooldown_line`). Size is now reduce of
  the number of powerlines of the grid.
- [BREAKING]  merge of `act.vars_action` into `env.attr_list_vect` which implemented the same concepts.
- [BREAKING] the runner now save numpy compressed array to lower disk usage. Previous saved runner are not compatible.
- [FIXED] `grid2op.PlotGrid` rounding error when casting from np.float32 to python.float
- [FIXED] `grid2op.BaseEnv.fast_forward_chronics` Calls the correct methods and is now working properly
- [FIXED] `__iadd__` is now properly implemented for the action with proper care given to action types.
- [UPDATED] MultiEnv now exchange only numpy arrays and not class objects.
- [UPDATED] Notebooks are updated to reflect API improvements changes
- [UPDATED] `grid2op.make` can now handle the download & caching of datasets
- [UPDATED] Test/Sample datasets provide datetime related files .info
- [UPDATED] Test/Sample datasets grid_layout.json
- [UPDATED] `grid2op.PlotGrid` Color schemes and optional infos displaying
- [UPDATED] `grid2op.Episode.EpisodeReplay` Improved gif output performance
- [UPDATED] Action and Observation are now created without having to call `init_grid(gridobject)` which lead to
  small speed up and memory saving.

[0.7.1] - 2020-04-22
----------------------
- [FIXED] a bug in the chronics making it not start at the appropriate time step
- [FIXED] a bug in "OneChangeThenNothing" agent that prevent it to be restarted properly.
- [FIXED] a bug with the generated docker file that does not update to the last version of the package.
- [FIXED] numpy, by default does not use the same datatype depending on the platform. We ensure that
  floating value are always `np.float32` and integers are always `np.int32`
- [ADDED] a method to extract only some part of a chronic.
- [ADDED] a method to "fast forward" the chronics
- [ADDED] class `grid2op.Reward.CombinedScaledReward`: A reward combiner with linear interpolation to stay within a
  given range.
- [ADDED] `grid2op.Reward.BaseReward.set_range`: All rewards have a default setter for their `reward_min` and
  `reward_max` attributes.
- [ADDED] `grid2op.PlotGrid`: Revamped plotting capabilities while keeping the interface we know from `grid2op.Plot`
- [ADDED] `grid2op.replay` binary: This binary is installed with grid2op and allows to replay a runner log with
  visualization and gif export
- [ADDED] a `LicensesInformation` file that put a link for all dependencies of the project.
- [ADDED] make multiple dockers, one for testing, one for distribution with all extra, and one "light"
- [UPDATED] test data and datasets are no longer included in the package distribution
- [UPDATED] a new function `make_new` that will make obsolete the "grid2op.download" script in future versions
- [UPDATED] the python "requests" package is now a dependency

[0.7.0] - 2020-04-15
--------------------
- [BREAKING] class `grid2op.Environment.BasicEnv` has been renamed `BaseEnv` for consistency. As this class
  should not be used outside of this code base, no backward compatibility has been enforced.
- [BREAKING] class `grid2op.Environment.ObsEnv` has been renamed `_ObsEnv` to insist on its "privateness". As this class
  should not be used outside of this code base, no backward compatibility has been enforced.
- [BREAKING] the "baselines" directory has been moved in another python package that will be released soon.
- [DEPRECATION] `grid2op.Action.TopoAndRedispAction` is now `grid2op.Action.TopologyAndDispatchAction`.
- [FIXED] Performances caveats regarding `grid2op.Backend.PandaPowerBackend.get_topo_vect`: Reduced the method running
  time and reduced number of direct calls to it.
- [FIXED] Command line install scripts: Can now use `grid2op.main` and `grid2op.download` after installing the package
- [FIXED] a bug that prevented to perform redispatching action if the sum of the action was neglectible (*eg* 1e-14)
  instead of an exact `0`.
- [FIXED] Manifest.ini and dockerfile to be complient with standard installation of a python package.
- [ADDED] a notebook to better explain the plotting capabilities of grid2op (work in progrress)
- [ADDED] `grid2op.Backend.reset` as a way for backends to implement a faster way to reload the grid. Implemented in
  `grid2op.Backend.PandaPowerBackend`
- [ADDED] `grid2op.Action.PowerlineChangeAndDispatchAction` A subset of actions to limit the agents scope to
  'switch line' and 'dispatch' operations only
- [ADDED] `grid2op.Action.PowerlineChangeAction` A subset of actions to limit the agents scope to 'switch line'
  operations only
- [ADDED] `grid2op.Action.PowerlineSetAndDispatchAction` A subset of actions to limit the agents scope to 'set line'
  and 'dispatch' operations only
- [ADDED] `grid2op.Action.PowerlineSetAction` A subset of actions to limit the agents scope to 'set line' operations
  only
- [ADDED] `grid2op.Action.TopologySetAction` A subset of actions to limit the agents scope to 'set' operations only
- [ADDED] `grid2op.Action.TopologySetAndDispatchAction` A subset of actions to limit the agents scope to 'set' and
  'redisp' operations only
- [ADDED] `grid2op.Action.TopologyChangeAction` A subset of actions to limit the agents scope to 'change' operations
  only
- [ADDED] `grid2op.Action.TopologyChangeAndDispatchAction` A subset of actions to limit the agents scope to 'change'
  and 'redisp' operations only
- [ADDED] `grid2op.Action.DispatchAction` A subset of actions to limit the agents scope to 'redisp' operations only
- [ADDED] a new method to plot other values that the default one for plotplotly.
- [ADDED] a better plotting utilities that is now consistent with `PlotPlotly`, `PlotMatplotlib` and `PlotPyGame`
- [ADDED] a class to replay a logger using `PlotPyGame` class (`grid2op.Plot.EpisodeReplay`)
- [ADDED] a method to parse back the observations with lower memory footprint and faster, when the observations
  are serialized into a numpy array by the runner, and only some attributes are necessary.
- [ADDED] fast implementation of "replay" using PlotPygame and EpisodeData
- [UPDATED] overall documentation: more simple theme, easier organization of each section.


[0.6.1] - 2020-04-??
--------------------
- [FIXED] `Issue #54 <https://github.com/rte-france/Grid2Op/issues/54>`_: Setting the bus for disconnected lines no
  longer counts as a substation operation.
- [FIXED] if no redispatch actions are taken, then the game can no more invalid a provided action due to error in the
  redispatching. This behavior was caused by increase / decrease of the system losses that was higher (in absolute
  value) than the ramp of the generators connected to the slack bus. This has been fixed by removing the losses
  of the powergrid in the computation of the redispatching algorithm. **side effect** for the generator connected
  to the slack bus, the ramp min / up as well as pmin / pmax might not be respected in the results data provided
  in the observation for example.
- [FIXED] a bug in the computation of cascading failure that lead (sometimes) to diverging powerflow when in the fact
  the powerflow did not diverge.
- [FIXED] a bug in the `OneChangeThenNothing` agent.
- [FIXED] a bug that lead to impossibility to load a powerline after a cascading failure in some cases. Now fixed by
  resetting the appropriate vectors when calling "env.reset".
- [FIXED] function `env.attach_render` that uses old names for the grid layout
- [ADDED] Remember last line buses: Reconnecting a line without providing buses will reconnect it to the buses it
  was previously connected to (origin and extremity).
- [ADDED] Change lines status (aka. switch_line_status) unitary actions for subclasses of AgentWithConverter.
- [ADDED] Dispatching unitary actions for subclasses of AgentWithConverter.
- [ADDED] CombinedReward. A reward combiner to compute a weighted sum of other rewards.
- [ADDED] CloseToOverflowReward. A reward that penalize agents when lines have almost reached max capacity.
- [ADDED] DistanceReward. A reward based on how far way from the original topology the current grid is.
- [ADDED] BridgeReward. A reward based on graph connectivity, see implementation in grid2op.Reward.BridgeReward for
  details

[0.6.0] - 2020-04-03
---------------------
- [BREAKING] `grid2op.GameRules` module renamed to `grid2op.RulesChecker`
- [BREAKING] `grid2op.Converters` module renamed `grid2op.Converter`
- [BREAKING] `grid2op.ChronicsHandler` renamed to `grid2op.Chronics`
- [BREAKING] `grid2op.PandaPowerBackend` is moved to `grid2op.Backend.PandaPowerBackend`
- [BREAKING] `RulesChecker.Allwayslegal` is now `Rules.Alwayslegal`
- [BREAKING] Plotting utils are now located in their own module `grid2op.Plot`
- [DEPRECATION] `HelperAction` is now called `ActionSpace` to better suit open ai gym name. Use of `HelperAction`
  will be deprecated in future versions.
- [DEPRECATION] `ObservationHelper` is now called `ObservationSpace` to better suit open ai gym name.
  Use of `ObservationHelper` will be deprecated in future versions.
- [DEPRECATION] `Action` class has been split into `BaseAction` that serve as an abstract base class for all
  action class, and `CompleteAction` (that inherit from BaseAction) for the class allowing to perform every
  modification implemented in grid2op.
- [DEPRECATION] `Observation` class has renamed `BaseObservation` that serve as an abstract base class for all
  observation classes. Name Observation will be deprecated in future versions.
- [DEPRECATION] `Agent` class has renamed `BaseAgent` that serve as an abstract base class for all
  agent classes. Name Agent will be deprecated in future versions.
- [DEPRECATION] `Reward` class has renamed `BaseReward` that serve as an abstract base class for all
  reward classes. Name Reward will be deprecated in future versions.
- [DEPRECATION] `LegalAction` class has renamed `BaseRules` that serve as an abstract base class for all
  type of rules classes. Name `LegalAction` will be deprecated in future versions.
- [DEPRECATION] typo fixed in `PreventReconection` class (now properly named `PreventReconnection`)
- [ADDED] different kind of "Opponent" can now be implemented if needed (missing deep testing, different type of
  class, and good documentation)
- [ADDED] implement other "rewards" to look at. It is now possible to have an environment that will compute more rewards
  that are given to the agent through the "information" return argument of `env.step`. See the documentation of
  Environment.other_rewards.
- [ADDED] Alternative method to load datasets based on new dataset format: `MakeEnv.make2`
- [ADDED] Layout of the powergrid is part of the `GridObject` and is serialized along with the
  action_space and observation_space. Plotting utilities no longer require specific layout (custom layout
  can still be provided)
- [ADDED] A new kind of actions that can change the value (and buses) to which shunt are connected. This support will
  be helpfull for the `VoltageControler` class.
- [FIXED] Loading L2RPN_2019 dataset
- [FIXED] a bug that prevents the voltage controler to be changed when using `grid2op.make`.
- [FIXED] `time_before_cooldown_line` vector were output twice in observation space
  (see `issue 47 <https://github.com/rte-france/Grid2Op/issues/47>`_ part 1)
- [FIXED] the number of active bus on a substation was not computed properly, which lead to some unexpected
  behavior regarding the powerlines switches (depending on current stats of powerline, changing the buses of some
  powerline has different effect)
  (see `issue 47 <https://github.com/rte-france/Grid2Op/issues/47>`_ part 2)
- [FIXED] wrong voltages were reported for PandapowerBackend that causes some isolated load to be not detected
  (see `issue 51 <https://github.com/rte-france/Grid2Op/issues/51>`_ )
- [FIXED] improve the install script to not crash when numba can be installed, but cannot be loaded.
  (see `issue 50 <https://github.com/rte-france/Grid2Op/issues/50>`_ )
- [UPDATED] import documentation of `Space` especially in case someone wants to build other type of Backend

[0.5.8] - 2020-03-20
--------------------
- [ADDED] runner now is able to show a progress bar
- [ADDED] add a "max_iter" in the runner.
- [ADDED] a repository in this github for the baseline (work in progress)
- [ADDED] include grid2Viz in a notebook (the notebook "StudyYourAgent")
- [ADDED] when a file is not present in the chronics, the chronics_handler behaves as if
  nothing changes. If no files at all are provided, it raises an error.
- [ADDED] possibility to change the controler for the generator voltage setpoints
  (See `VoltageControler` for more information). It can be customized as of now.
- [ADDED] lots of new tests for majority of classes (ChronicsHandler, BaseAction, Observations etc.)
- [FIXED] voltages are now set to 0 when the powerline are disconnected, instead of being set to Nan in
  pandapower backend.
- [FIXED] `ReadPypowNetData` does not crash when argument "chunk_size" is provided now.
- [FIXED] some typos in the Readme
- [FIXED] some redispatching declared illegal but are in fact legal (due to
  a wrong assessment) (see `issue 44 <https://github.com/rte-france/Grid2Op/issues/44>`_)
- [FIXED] reconnecting a powerline now does not count the mandatory actions on both its ends (previously you could not
  reconnect a powerline with the L2RPN 2019 rules because it required acting on 2 substations) as "substation action"
- [UPDATED] add a blank environment for easier use.
- [UPDATED] now raise an error if the substations layout does not match the number of substations on the powergrid.
- [UPDATED] better handling of system without numba `issue 42 <https://github.com/rte-france/Grid2Op/issues/42>`_)
- [UPDATED] better display of the error message if all dispatchable generators are set
  `issue 39 <https://github.com/rte-france/Grid2Op/issues/39>`_
- [UPDATED] change the link to the doc in the notebook to point to readthedoc and not to local documentation.
- [UPDATED] Simulate action behavior result is the same as stepping given perfect forecasts at t+1 

[0.5.7] - 2020-03-03
--------------------
- [ADDED] a new environment with consistant voltages based on the case14 grid of pandapower (`case14_relistic`)
- [ADDED] a function to get the name on the element of the graphical representation.
- [ADDED] a new class to (PlotMatPlotlib) to display the grid layout and the position of the element,
  as well as their name and ID
- [ADDED] possibility to read by chunk the data (memory efficiency and huge speed up at the beginning of training)
  (`issue 21 <https://github.com/rte-france/Grid2Op/issues/21>`_)
- [ADDED] improved method to limit the episode length in chronics handler.
- [ADDED] a method to project some data on the layout of the grid (`GetLayout.plot_info`)
- [FIXED] a bug in the simulated reward (it was not initialized properly)
- [FIXED] add the "prod_charac.csv" for the test environment `case14_test`, `case14_redisp`, `case14_realistic` and
  `case5_example`
- [FIXED] fix the display bug in the notebook of the l2rpn starting kit with the layout of the 2 buses
- [UPDATED] now attaching the layout metadata directly into the environment
- [UPDATED] `obs.simulate` now has the same code as `env.step` this include the same signature and the
  possibility to simulate redispatching actions as well.
- [UPDATED] Notebook 6 to train agent more efficiently (example: prediction of actions in batch)
- [UPDATED] PlotGraph to derive from `GridObjects` allowing to be inialized at creation and not when first
  observation is loaded (usable without observation)
- [UPDATED] new default environment (`case14_realistic`)
- [UPDATED] data for the new created environment.
- [UPDATED] implement redispatching action in `obs.simulate`
- [UPDATED] refactoring `Environment` and `ObsEnv` to inherit from the same base class.

[0.5.6] - 2020-02-25
--------------------
- [ADDED] Notebook 6 to explain multi environment
- [ADDED] more type of agents in the notebook 3
- [FIXED] Environment now properly built in MultiEnvironment
- [FIXED] Notebook 3 to now work with both neural network
- [FIXED] remove the "print" that displayed the path of the data used in MultiEnvironment
- [UPDATED] the action space for "IdToAct" now reduces the number of possible actions to only actions that don't
  directly cause a game over.

[0.5.5] - 2020-02-14
---------------------
- [ADDED] a easier way to set the thermal limits directly from the environment (`env.set_thermal_limit`)
- [ADDED] a new environment with redispatching capabilities (`case14_redisp`) including data
- [ADDED] a new convenient script to download the dataset, run `python3 -m grid2op.download --name "case14_redisp"`
  from the command line.
- [ADDED] new rewards to better take into account redispatching (`EconomicReward` and `RedispReward`)
- [ADDED] a method to check if an action is ambiguous (`act.is_ambiguous()`)
- [ADDED] a method to set more efficiently the id of the chronics used in the environment (`env.set_id`)
- [ADDED] env.step now propagate the error in "info" output (but not yet in  `obs.simulate`)
- [ADDED] notebooks for redispatching (see `getting_started/5_RedispacthingAgent.ipynb`)
- [ADDED] now able to initialize a runner from a valid environment (see `env.get_params_for_runner`)
- [FIXED] reconnecting too soon a powerline is now forbidden in l2rpn2019 (added the proper legal action)
- [UPDATED] more information in the error when plotly and seaborn are not installed and trying to load the
  graph of the grid.
- [UPDATED] setting an object to a busbar higher (or equal) than 2 now leads to an ambiguous action.
- [UPDATED] gitignore to really download the "prod_charac.csv" file
- [UPDATED] private member in action space and observation space (`_template_act` and `_empty_obs`)
  to make it clear it's not part of the public API.
- [UPDATED] change default environment to `case14_redisp`
- [UPDATED] notebook 2 now explicitely says the proposed action is ambiguous in a python cell code (and not just
  in the comments) see issue (`issue 27 <https://github.com/rte-france/Grid2Op/issues/27>`_)

[0.5.4] - 2020-02-06
---------------------
- [ADDED] better handling of serialization of scenarios.

[0.5.3] - 2020-02-05
---------------------
- [ADDED] parrallel processing of the environment: evaluation in parrallel of the same agent in different environments.
- [ADDED] a way to shuffle the order in which different chronics are read from the hard drive (see MultiFolder.shuffle)
- [FIXED] utility script to push docker file
- [FIXED] some tests were not passed on the main file, because of a file ignore by git.
- [FIXED] improve stability of pandapower backend.
- [UPDATED] avoid copying the grid to build observation


[0.5.2] - 2020-01-27
---------------------
- [ADDED] Adding a utility to retrieve the starting kit L2RPN 2019 competition.
- [ADDED] Layout of the powergrid graph of the substations for both the
  `5bus_example` and the `CASE_14_L2RPN2019`.
- [FIXED] Runner skipped half the episode in some cases (sequential, even number of scenarios). Now fixed.
- [FIXED] Some typos on the notebook "getting_started\4-StudyYourAgent.ipynb".
- [FIXED] Error in the conversion of observation to dictionary. Twice the same keys were used
  ('time_next_maintenance') for both `time_next_maintenance` and `duration_next_maintenance`.
- [UPDATED] The first chronics that is processed by a runner is not the "first" one on the hardrive
  (if sorted in alphabetical order)
- [UPDATED] Better layout of substation layout (in case of multiple nodes) in PlotGraph

[0.5.1] - 2020-01-24
--------------------
- [ADDED] extra tag 'all' to install all optional dependencies.
- [FIXED] issue in the documentation of BaseObservation, voltages are given in kV and not V.
- [FIXED] a bug in the runner that prevented the right chronics to be read, and output wrong names
- [FIXED] a bug preventing import if plotting packages where not installed, that causes the documentation to crash.

[0.5.0] - 2020-01-23
--------------------
- [BREAKING] BaseAction/Backend has been modified with the implementation of redispatching. If
  you used a custom backend, you'll have to implement the "redispatching" part.
- [BREAKING] with the introduction of redispatching, old action space and observation space,
  stored as json for example, will not be usable: action size and observation size
  have been modified.
- [ADDED] A converter class that allows to pre-process observation, and post-process action
  when given to an `BaseAgent`. This allows for more flexibility in the `action_space` and
  `observation_space`.
- [ADDED] Adding another example notebook `getting_started/Example_5bus.ipynb`
- [ADDED] Adding another renderer for the live environment.
- [ADDED] Redispatching possibility for the environment
- [ADDED] More complete documentation of the representation of the powergrid
  (see documentation of `Space`)
- [FIXED] A bug in the conversion from pair unit to kv in pandapower backend. Adding some tests for that too.
- [UPDATED] More complete documentation of the BaseAction class (with some examples)
- [UPDATED] More unit test for observations
- [UPDATED] Remove the TODO's already coded
- [UPDATED] GridStateFromFile can now read the starting date and the time interval of the chronics.
- [UPDATED] Documentation of BaseObservation: adding the units
  (`issue 22 <https://github.com/rte-france/Grid2Op/issues/22>`_)
- [UPDATED] Notebook `getting_started/4_StudyYourAgent.ipynb` to use the converter now (much shorter and clearer)

[0.4.3] - 2020-01-20
--------------------
- [FIXED] Bug in L2RPN2019 settings, that had not been modified after the changes of version 0.4.2.

[0.4.2] - 2020-01-08
--------------------
- [BREAKING] previous saved BaseAction Spaces and BaseObservation Spaces (as dictionary) are no more compatible
- [BREAKING] renaming of attributes describing the powergrid across classes for better consistency:

=============================    =======================  =======================
Class Name                       Old Attribute Name       New Attribute Name
=============================    =======================  =======================
Backend                           n_lines                  n_line
Backend                           n_generators             n_gen
Backend                           n_loads                  n_load
Backend                           n_substations            n_sub
Backend                           subs_elements            sub_info
Backend                           name_loads               name_load
Backend                           name_prods               name_gen
Backend                           name_lines               name_line
Backend                           name_subs                name_sub
Backend                           lines_or_to_subid        line_or_to_subid
Backend                           lines_ex_to_subid        line_ex_to_subid
Backend                           lines_or_to_sub_pos      line_or_to_sub_pos
Backend                           lines_ex_to_sub_pos      line_ex_to_sub_pos
Backend                           lines_or_pos_topo_vect   line_or_pos_topo_vect
Backend                           lines_ex_pos_topo_vect   lines_ex_pos_topo_vect
BaseAction / BaseObservation     _lines_or_to_subid       line_or_to_subid
BaseAction / BaseObservation     _lines_ex_to_subid       line_ex_to_subid
BaseAction / BaseObservation     _lines_or_to_sub_pos     line_or_to_sub_pos
BaseAction / BaseObservation     _lines_ex_to_sub_pos     line_ex_to_sub_pos
BaseAction / BaseObservation     _lines_or_pos_topo_vect  line_or_pos_topo_vect
BaseAction / BaseObservation     _lines_ex_pos_topo_vect  lines_ex_pos_topo_vect
GridValue                        n_lines                  n_line
=============================    =======================  =======================

- [FIXED] Runner cannot save properly action and observation (sizes are not computed properly)
  **now fixed and unit test added**
- [FIXED] Plot utility has a bug in extracting grid information.
  **now fixed**
- [FIXED] gym compatibility issue for environment
- [FIXED] checking key-word arguments in "make" function: if an invalid argument is provided,
  it now raises an error.
- [UPDATED] multiple random generator streams for observations
- [UPDATED] Refactoring of the BaseAction and BaseObservation Space. They now both inherit from "Space"
- [UPDATED] the getting_started notebooks to reflect these changes

[0.4.1] - 2019-12-17
--------------------
- [FIXED] Bug#14 : Nan in the observation space after switching one powerline [PandaPowerBackend]
- [UPDATED] plot now improved for buses in substations

[0.4.0] - 2019-12-04
--------------------
- [ADDED] Basic tools for plotting with the `PlotPlotly` module
- [ADDED] support of maintenance operation as well as hazards in the BaseObservation (and appropriated tests)
- [ADDED] support for maintenance operation in the Environment (read from the chronics)
- [ADDED] example of chronics with hazards and maintenance
- [UPDATED] handling of the `AmbiguousAction` and `IllegalAction` exceptions (and appropriated tests)
- [UPDATED] various documentation, in particular the class BaseObservation
- [UPDATED] information retrievable `BaseObservation.state_of`

[0.3.6] - 2019-12-01
--------------------
- [ADDED] functionality to restrict action based on previous actions
  (impacts `Environment`, `RulesChecker` and `Parameters`)
- [ADDED] tests for the notebooks in `getting_started`
- [UPDATED] readme to properly show the docker capability
- [UPDATED] Readme with docker

[0.3.5] - 2019-11-28
--------------------
- [ADDED] serialization of the environment modifications
- [ADDED] the changelog file
- [ADDED] serialization of hazards and maintenance in actions (if any)
- [FIXED] error messages in `grid2op.GridValue.check_validity`
- [UPDATED] notebook `getting_started/4_StudyYourAgent.ipynb` to reflect these changes
