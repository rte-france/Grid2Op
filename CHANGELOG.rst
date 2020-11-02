Change Log
===========

[TODO]
--------------------
- [???] Extensive tests for BridgeReward
- [???] Extensive tests for DistanceReward
- [???] in the observation, make the possibility to retrieve the "active flow graph" (ie graph with edges having active
  flows, and nodes the active production / consumption) and "reactive flow graph"
- [???] better logging
- [???] add a "plot action" method
- [???] simulate in MultiEnv
- [???] in MultiEnv, when some converter of the observations are used, have each child process to compute
  it in parallel and transfer the resulting data.
- [???] "asynch" multienv
- [???] properly model interconnecting powerlines
- [???] model curtailment
- [???] model batteries / pumped storage in grid2op (generator but that can be charged / discharged)
- [???] model dumps (as in dump storage) in grid2op (stuff that have a given energy max, and cannot produce more than
  the available energy)

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
  would not load it properly.
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
  the "why" the action is illegal is displayed (instead of a generic "this action is illegal".
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
  `grid2op.chronics.MultifolderWithCache` (related to `Issued #98 <https://github.com/rte-france/Grid2Op/issues/98>`_)
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
- [UPDATED] new default environment (`case14_relistic`)
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
- [FIXED] Error in the conversion of observation to dictionnary. Twice the same keys were used
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
- [BREAKING] previous saved BaseAction Spaces and BaseObservation Spaces (as dictionnary) are no more compatible
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
