# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

grid2op is a gymnasium-compatible environment for sequential decision making on power grids (RL research, the
L2RPN competitions). Pure Python, MPL-2.0, an LF Energy project. The package is `grid2op/`; everything else at the
root is docs, notebooks (`getting_started/`), CI, or governance. The version lives only in `grid2op/__init__.py`
(`__version__`, currently a `.dev0`), and `pyproject.toml` reads it dynamically.

## Python and environments

- There is no bare `python` on this machine, only `python3`. Prefer a repo venv interpreter (all untracked):
  `venv/bin/python` (3.13, has lightsim2grid 0.12 and gymnasium, the most complete), `venv_python39/bin/python`
  (3.9, lightsim2grid 0.6.0), `venv_314/bin/python` (3.14, no optional deps). None of them has pytest.
- Dev install: `pip install -e .[test]` (what CI uses; brings lightsim2grid, gymnasium, numba, nbconvert, plotting
  libs) or `pip install -e .[optional,docs]`. Several tests import lightsim2grid unconditionally, so use `[test]`.
  Python 3.8 needs `pyproject_38.toml` copied over `pyproject.toml`.
- `grid2op.create_test_suite` (used by external backends) only exists with an editable install from source.

## Tests

The suite uses the standard library `unittest` only. There is no pytest configuration and pytest is not
installed; if you install it, expect noise from `Test*`-named mixin classes that are not `TestCase`s (e.g.
`TestIADD` in `test_Action.py`) and treat unittest as the reference. Tests live in `grid2op/tests/` and can be run
from the repo root or from `grid2op/tests/` (`helper_path_test.py` derives data paths from `__file__`).
CI always exports `_GRID2OP_FORCE_TEST=1`, which forces `test=True` in every `grid2op.make()` so nothing is ever
downloaded; do the same.

```bash
cd grid2op/tests
export _GRID2OP_FORCE_TEST=1
python3 -m unittest discover                                        # full suite: very long, CI shards it over ~9 jobs
python3 -m unittest test_Action                                     # one module
python3 -m unittest test_Parameters.TestParameters.test_default_builds   # one method
python3 -m unittest test_Action -k redisp                           # substring filter
python3 -m unittest test_issue_*                                    # all regression tests
python3 -m unittest grid2op.tests.test_Parameters                   # same thing, from the repo root
coverage run -m unittest test_Action && coverage report -m          # .coveragerc omits data/ and the tests themselves
```

- Regression tests for GitHub issues are `grid2op/tests/test_issue_<number>.py`, one per issue.
- Standard test shape: `unittest.TestCase` with `grid2op.make(...)` wrapped in
  `with warnings.catch_warnings(): warnings.filterwarnings("ignore")`, `_add_to_name=type(self).__name__` (so the
  dynamically generated classes of each test class get unique names and never collide; only honoured with
  `test=True`), and `self.env.close()` in `tearDown`. Shared behaviour lives in non-TestCase `Base*` mixins that
  concrete tests combine as `class TestX(BaseX, unittest.TestCase)` with a `make_backend()` method.
- CI layout (`.circleci/config.yml`): named groups (agent, converter, runner/episode/score, env general,
  alert/alarm, time series, backend, `test_issue_*`) each run in their own job; everything else is listed by
  `grid2op/tests/helper_list_test.py` and split across the `test_generic` shards. That script skips the modules in
  its `li_tested_elsewhere` list, so a module moved into a dedicated CI job must also be added there or it runs twice.
- `test_notebooks_getting_started.py` executes the `getting_started/` notebooks. Clear notebook outputs before
  committing (`jupyter nbconvert --clear-output --inplace <nb>.ipynb`).
- Test fixtures: `grid2op/data_test/` (5bus_example variants, chronics_with_*, multimix, ...) reached through
  `helper_path_test.PATH_DATA_TEST`, loaded as `grid2op.make(os.path.join(PATH_DATA_TEST, "5bus_example"),
  test=True, ...)`; `grid2op/data/` holds the small shipped copies of the real environments that
  `make("<name>", test=True)` loads.
- Backend conformance suites for external backends (lightsim2grid, pypowsybl2grid) are built by
  `grid2op.create_test_suite` from `AAATestBackendAPI` (`aaa_test_backend_interface.py`, `test_00`..`test_40` on
  `data/educ_case14_storage`) plus the `Base*` mixins in `BaseBackendTest.py`, `BaseRedispTest.py` and
  `test_Environment.py`. `helper_path_test.MakeBackend.make_backend_with_glue_code` shows the reset dance needed
  between backend tests (`Backend._clear_class_attribute()`, `set_env_name`, `set_n_busbar_per_sub`,
  `set_detachment_is_allowed`) because grid data is stored on the classes, not the instances (see Architecture).

## Docs

```bash
pip install -e .[docs]
make html                                   # or: sphinx-build -b html docs documentation
```

Output lands in `documentation/html/index.html` (`documentation/` is gitignored). Docstrings are Sphinx/reST
(numpydoc); see "Comments and docstrings" below for what must never appear in them. A new public API must be
added to the matching `docs/user/*.rst`. Extension guides (writing a backend, building an environment folder,
observation internals) are in `docs/grid2op_extend/`; multiprocessing/class pickling problems are covered in
`docs/troubleshoot.rst`.

## Architecture

### Grid data lives on classes, not instances

Nearly everything (`BaseAction`, `BaseObservation`, the spaces, `Backend`, `_BackendAction`, `BaseEnv`) inherits
`grid2op.Space.GridObjects`. The grid description (`n_sub`, `n_line`, `name_load`, `*_to_subid`, `*_to_sub_pos`,
`*_pos_topo_vect`, generator characteristics, `n_busbar_per_sub`, `detachment_is_allowed`,
`shunts_data_available`, `glop_version`, ...) is stored as **class attributes**, so all actions/observations of one
env share it and vector sizes are fixed per class. `GridObjects.init_grid(gridobj)` creates an env-specific
subclass on the fly named `<Cls>_<env_name>` (with suffixes such as `_<glop_version>`, `_noshunt`, `_<n_busbar>`,
`_allowDetach`) and caches it in the `GridObjects` module globals. Since 1.11 the backend class name is appended
to the env name, giving names like `PlayableAction_l2rpn_case14_sandboxPandaPowerBackend`. Read grid data through
the class: `cls = type(self); cls.n_line`. The un-suffixed base classes are wiped after init
(`_clear_class_attribute`, `_clear_grid_dependant_class_attributes`), and backend instance attributes that shadow
class attributes are deleted after `assert_grid_correct`.

Consequences:

- **Pickling / multiprocessing.** `__reduce__` ships the class dict and `init_grid_from_dict_for_pickle` rebuilds
  it, but classes referenced by name in spawned processes fail with "Can't get attribute 'ActionSpace_...' on
  module grid2op.Space.GridObjects". Two fixes, not to be combined: `env.generate_classes()` writes the classes as
  `.py` files into `<env_dir>/_grid2op_classes/` and `make(..., experimental_read_from_local_dir=True)` imports
  them; or `make(..., class_in_file=True)` writes them into a temporary subdirectory that is removed on `close()`.
- **Backward compatibility.** `glop_version` plus `process_grid2op_compat` (overridden in `BaseAction` and
  `BaseObservation`) strips newer features (storage, alarms, alerts, `n_busbar`, detachment) so old `EpisodeData`
  can be read back. It is driven by the test-only `make` kwarg `_compat_glop_version`.

### Environment creation: `grid2op.make`

`make(dataset, **kwargs)` (`grid2op/MakeEnv/Make.py`, keyword-only kwargs validated against `ERR_MSG_KWARGS`)
resolves the dataset in this order: `_GRID2OP_FORCE_TEST` forces `test=True`; an existing path is loaded directly;
`test=True` with a name loads the small bundled copy in `grid2op/data/<name>` (names in
`MakeEnv/_aux_var.py::TEST_DEV_ENVS`); a name present in `~/data_grid2op` (overridable with `data_path` in
`~/.grid2opconfig.json`, see `get_current_local_dir` / `change_local_dir`) is used; otherwise it is downloaded
there. A `.multimix` marker file turns the directory into a `MultiMixEnvironment`.

`make_from_dataset_path` (`MakeFromPath.py`) calls `get_default_env_kwargs`, which **executes the dataset's
`config.py`** (it must define a dict `config`). Its keys (`backend`, `action_class`, `observation_class`,
`reward_class`, `gamerules_class`, `chronics_class`, `grid_value_class`, `data_feeding_kwargs`,
`voltagecontroler_class`, `names_chronics_to_grid`, `thermal_limits`, `other_rewards`, `opponent_*`, ...) are
overridden by `make` kwargs. `Parameters` never come from `config.py`: they come from `difficulty_levels.json`
(kwarg `difficulty`, else the `"competition"` level, else the highest one) or `parameters.json`. The grid file is
`grid.<ext>` for an ext in `backend.supported_grid_format`; other files read are `grid_layout.json`,
`prods_charac.csv`, `storage_units_charac.csv`, `alerts_info.json` and an optional `grid_forecast.json` (a
separate grid used only by `simulate`).

`Environment._init_backend` then: sets env name, busbar count, shunt and detachment flags on the backend class and
calls `load_grid_public`; loads storage/redispatch/layout/alert data; `backend.assert_grid_correct()` (creates the
env-specific backend class); swaps the env's own class to `Environment_<name>` and builds the `_BackendAction`
class; creates the `RulesChecker` and runs `init_grid` on the action/observation classes and spaces; initializes
the chronics with the grid names, then `RewardHelper`, voltage controller and opponent; finally performs one
internal do-nothing step with overflow disconnection turned off.

### One `env.step(action)` (`BaseEnv.step` in `Environment/baseEnv.py`; `Environment` does not override it)

1. An ambiguous action (`is_ambiguous()`) or a failing `backend_dependant_callback` is replaced by do-nothing and
   flagged `is_ambiguous`; a failing `check_reconnection_valid` makes it illegal.
2. Legality: `self._game_rules(action, env)` (`RulesChecker`). Illegal actions become do-nothing with `is_illegal`.
3. `_update_actions()` advances the chronics (`chronics_handler.next_time_step()`) into `_env_modification`, a
   `CompleteAction` carrying injections, maintenance, hazards and `prod_v`. `StopIteration` ends the episode.
4. Injections: new prod setpoints, `_compute_storage`, curtailment, `_backend_action += action` (with redispatch
   and storage zeroed), `_aux_apply_detachment`.
5. Redispatch (`_aux_apply_redisp`): `_prepare_redisp` cancels invalid requests (`IllegalRedispatching`), the
   `FeasibilityGuard` limits storage / curtailment if `LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION`, then
   `_make_redisp` asks the redispatch solver (`Environment/dispatch/`, `DefaultRedispatchSolver` = scipy SLSQP,
   replaceable with the `redispatch_solver` make kwarg) for a dispatch compensating storage, curtailment and
   detachment within pmin/pmax/ramps; failure raises `ImpossibleRedispatching` and ends the episode. Storage,
   curtailment and detachment are computed by modules of the same package, on a `RedispatchState` that the
   `BaseEnv` attributes (`_actual_dispatch`, `_storage_power`, ...) are properties of.
6. `_backend_action += _env_modification`, actual dispatch and storage are set, then the voltage-controller action
   and the opponent attack (`_aux_handle_attack` -> `OpponentSpace.attack`) are added.
7. `backend.apply_action_public(_backend_action)`; `ImpossibleTopology` / `BackendError` end the episode.
8. `_aux_run_pf_after_state_properly_set` -> `backend.next_grid_state`: `runpf` (wrapped by
   `_runpf_with_diverging_exception`, which also fails on isolated loads/gens when detachment is not allowed), then
   the protection loop: lines above `HARD_OVERFLOW_THRESHOLD` x limit, or above `SOFT_OVERFLOW_THRESHOLD` x limit
   for more than `NB_TIMESTEP_OVERFLOW_ALLOWED` steps, are disconnected and the power flow re-run until stable
   (skipped with `NO_OVERFLOW_DISCONNECTION`). On success `_aux_register_env_converged` updates counters,
   cooldowns, maintenance and topology, and `get_obs()` -> `ObservationSpace.__call__` -> `obs.update(env)`.
9. Reward from `RewardHelper` plus `other_rewards`; `done = has_error or is_done or chronics_handler.done()`; on
   error the observation goes through `set_game_over`. **Bad actions never raise**: everything is reported in
   `info["exception"]` (a list) and the flags `is_illegal`, `is_ambiguous`, `failed_redispatching`,
   `is_illegal_reco`, `disc_lines`, `opponent_attack_*`, `time_series_id`, `rewards`.

`_BackendAction` (`Action/_backendAction.py`) accumulates changes in `ValueStore`s (values plus a `changed` mask;
only changed entries are meaningful, so backends apply deltas). Its `__iadd__` merges in the order detach ->
injections -> redispatch -> storage -> shunts -> line status -> change_bus -> set_bus -> reconcile line ends.
Backends read it via `backend_action()` -> `(active_bus, (prod_p, prod_v, load_p, load_q, storage), topo, shunts)`
and `get_*_bus()`; `last_topo_registered` remembers buses so a reconnected line returns to its previous bus.

`reset(options=...)` accepts the keys in `BaseEnv.KEYS_RESET_OPTIONS` ("time serie id", "init state", "init ts",
"max step", "thermal limit", "init datetime"). `change_parameters`, `change_reward` and
`change_forecast_parameters` only take effect at the next `reset()`; `env.parameters` returns a deepcopy.

### simulate and forecasts

`ObservationSpace` owns `obs_env`, an `_ObsEnv` (`Environment/_obsEnv.py`, itself a `BaseEnv`) whose backend is
`copy_public()` of the env backend, or a separate one when `observation_backend_*` / `grid_forecast.json` is given;
if the copy fails, `simulate` is disabled. `obs.simulate(act, time_step)` runs a normal step in it with the
forecast `Parameters` (legality is checked; `MAX_SIMULATE_PER_STEP/EPISODE` enforced). `obs.get_forecast_env()`
returns a `ForecastEnv` with `FromNPY` chronics built from the forecasts. `obs.get_simulator()` returns
`simulator.Simulator`, whose `predict()` calls chain and ignore time, cooldowns, the opponent and protections.

### Backend contract (`Backend/backend.py`)

Abstract methods: `load_grid`, `apply_action`, `runpf` (returns `(converged, exception)`), `get_topo_vect`,
`generators_info`, `loads_info`, `lines_or_info`, `lines_ex_info`. The env only calls the `*_public` wrappers
(`load_grid_public`, `apply_action_public`, `reset_public`, `copy_public`). `load_grid` must call
`can_handle_more_than_2_busbar()` or `cannot_handle_more_than_2_busbar()`, and `can_handle_detachment()` or
`cannot_handle_detachment()`, otherwise grid2op warns and falls back to 2 busbars and no detachment. `shunt_info`
is required when `shunts_data_available`; `reset`, `copy`, `close`, `get_line_status`, `get_line_flow`,
`_disconnect_line`, `storages_info`, `get_theta`, `get_class_added_name` are optional. `assert_grid_correct`
swaps the instance to the env-specific class and attaches `my_bk_act_class` / `_complete_action_class`.
`PandaPowerBackend` is the production backend; `Backend/educPandaPowerBackend.py` is a readable teaching example
(no shunts/storage, not exported); `Converter/BackendConverter.py` keeps a source backend's names and ordering
while a target backend computes. Guide: `docs/grid2op_extend/createbackend.rst`.

### Actions, observations, spaces

Action classes differ only by `authorized_keys` / `attr_list_vect`: `BaseAction` allows everything (including
injections, hazards, maintenance); `CompleteAction` is the full set used by the env, chronics, opponent and voltage
controller; `PlayableAction` is the agent subset (no injection/hazards/maintenance); narrower ones are
`TopologyAction`, `PowerlineSetAction`, `DispatchAction`, `*AndDispatch`, `VoltageOnlyAction` and `DontAct` (no
keys; the default opponent action). `PlayableAction.update` warns on and ignores unknown keys; unauthorized
modifications make the action illegal at the ambiguity check. `detach_*` keys are removed from the class unless
`allow_detachment=True`.

`topo_vect` is one int per element (`dim_topo`), grouped by substation, indexed through `*_pos_topo_vect`:
`set_bus` 0 = no change, -1 = disconnect, 1..`n_busbar_per_sub` = busbar; `change_bus` is a boolean toggle;
`set_line_status` -1/0/+1 (reconnection restores each end's last bus); in observations -1 means disconnected.
`redispatch` is a MW delta (the env computes `actual_dispatch`), `curtail` a fraction of pmax (-1 = none),
`set_storage` MW with positive = charging, `detach_*` sets topo to -1 and zeroes the injection.

`SerializableActionSpace` (`sample`, `from_vect`, `get_all_unitary_*`, JSON; needs no env) is extended by
`ActionSpace` (adds `legal_action` and `__call__(dict, check_legal, env)`); `ObservationSpace` adds `obs_env`, a
reward helper and simulate counters. The env holds `_action_space` (the agent's class) and `_helper_action_env`
(`CompleteAction`).

### Chronics (time series)

`ChronicsHandler` wraps one `GridValue` (`initialize`, `load_next`, `check_validity`, `next_chronics`) and forwards
attribute access to it. `Multifolder` iterates the sorted `chronics/*` subdirectories, one `gridvalueClass` per
episode. `GridStateFromFile` reads `;`-separated `load_p`, `load_q`, `prod_p`, `prod_v`, `maintenance`, `hazards`
(`.csv` or `.csv.bz2`) plus `start_datetime.info` and `time_interval.info`; a missing file means that quantity is
never modified. `GridStateFromFileWithForecasts` adds `*_forecasted` files (multi-horizon via `h_forecast`). CSV
headers are object names, mapped with `names_chronics_to_backend = {"loads"|"prods"|"lines"|"subs":
{csv_name: backend_name}}` (identity if absent, bijection enforced). `FromNPY` takes arrays already in backend
order; `FromHandlers` (`Chronics/handlers/`) uses one handler per quantity; also `ChangeNothing`,
`FromOneEpisodeData`, `FromMultiEpisodeData`, `FromChronix2grid`, `MultifolderWithCache`. Chronics also provide the
initial grid state (`get_init_action`).

### Other layers

- **Rules**: `BaseRules.__call__(action, env) -> (bool, exc)`. `DefaultRules` = `LookParam`
  (`MAX_LINE_STATUS_CHANGED`, `MAX_SUB_CHANGED`, `can_use_simulate`) + `PreventReconnection` (cooldowns) +
  `PreventDiscoStorageModif`. `RulesByArea` is passed as an instance in the idf_2023 config.
- **Reward**: `BaseReward.initialize` sets `reward_min`/`reward_max`; `__call__(action, env, has_error, is_done,
  is_illegal, is_ambiguous)`. `RewardHelper` accepts a class or an instance; `other_rewards` land in
  `info["rewards"]`.
- **Opponent**: `BaseOpponent.attack(obs, agent_act, env_act, budget, previous_fails) -> (attack, duration)`;
  `OpponentSpace.attack` adds `budget_per_timestep` each step, rejects attacks with `duration * cost > budget`
  and enforces max duration and cooldown. The default `DontAct` + `NeverAttackBudget` means no opponent.
- **Parameters** (`grid2op/Parameters.py`): protections (`NO_OVERFLOW_DISCONNECTION`,
  `NB_TIMESTEP_OVERFLOW_ALLOWED`, `NB_TIMESTEP_RECONNECTION`, `HARD_/SOFT_OVERFLOW_THRESHOLD`), action limits and
  cooldowns (`NB_TIMESTEP_COOLDOWN_LINE/SUB`, `MAX_SUB_CHANGED`, `MAX_LINE_STATUS_CHANGED`,
  `MAX_SIMULATE_PER_STEP/EPISODE`), generators/storage (`IGNORE_MIN_UP_DOWN_TIME`, `ALLOW_DISPATCH_GEN_SWITCH_OFF`,
  `INIT_STORAGE_CAPACITY`, `ENV_DOES_REDISPATCHING`, `STOP_EP_IF_GEN_BREAK_CONSTRAINTS`), `ENV_DC`,
  `IGNORE_INITIAL_STATE_TIME_SERIE`.
- **Runner** (`Runner/runner.py`): `Runner(**env.get_params_for_runner(), agentClass=... | agentInstance=...)`
  then `.run(...)`; it recreates envs from kwargs and parallelizes with `Pool.starmap` (not supported on
  Windows). Output is `EpisodeData` (one directory per episode: `actions.npz`, `observations.npz`,
  `episode_meta.json`) or `CompactEpisodeData` (`use_compact_episode_data=True`, one `.npz` per episode).
- **gym_compat**: `gym_compat/utils.py` detects gym vs gymnasium; each class is a private `__Aux*` implementation
  mixed with a gym or gymnasium base via `type(...)`, and the un-suffixed names (`GymEnv`) point to the gymnasium
  variant when installed, else `GymEnv_Legacy` / `GymEnv_Modern`. `GymnasiumEnv` copies the env and drops
  `obs_env` unless `with_forecast=True`; `truncated` is always False. Spaces: `GymActionSpace` /
  `GymObservationSpace` (Dict), `BoxGymActSpace` / `BoxGymObsSpace`, `DiscreteActSpace` (built from
  `get_all_unitary_*`), `MultiDiscreteActSpace`.
- **MultiMixEnvironment**: dict-like set of `Environment` mixes; the first mix generates the classes and the
  others reuse them; `reset()` cycles through mixes (or `random=True`) and `__getattr__` forwards to the current
  one. Other env flavours: `TimedOutEnvironment`, `MaskedEnvironment`, `SingleEnvMultiProcess`,
  `MultiEnvMultiProcess`, `ForecastEnv`.
- **Exceptions** (`grid2op/Exceptions/`): root `Grid2OpException(RuntimeError)`; `EnvError`
  (`ImpossibleRedispatching`, `IncorrectNumberOf*`), `BackendError` (`DivergingPowerflow`, `IslandedGrid`,
  `ImpossibleTopology`, `DisconnectedLoad`, `DisconnectedGenerator`), `AmbiguousAction` (`InvalidLineStatus`,
  `InvalidRedispatching`, ...), `IllegalAction`, `ChronicsError`, `OpponentError`, `NoForecastAvailable`,
  `SimulateUsedTooMuch*`, `UnknownEnv`.

## Conventions (from CONTRIBUTING.md and the repo)

- Branches: work on a topic branch off the active `dev_X.Y.Z` branch (currently `dev_1.12.6`) and open PRs
  against it. `master` only receives release merges.
- Every commit must carry a human DCO signoff (`git commit -s`; see "Commits" below). The `commit-msg` hook
  (`scripts/check_dco_msg.py`) is installed in this clone and rejects unsigned commits; the `pre-commit` hook
  runs detect-secrets against `.secrets.baseline` and check-mailmap.
- Every user-facing change gets a line in `CHANGELOG.rst` under the top version section (`[1.12.6] - 2026-xx-yy`),
  tagged `[FIXED]`, `[ADDED]`, `[IMPROVED]`, `[BREAKING]`, `[UPDATED]` or `[DEPRECATION]`. The "Work in progress"
  block above it is the maintainers' wishlist, not a release section.
- Bug fixes ship with a test that fails before the fix; tests must be deterministic (seed RNGs) and offline.
- Every `.py` file starts with the MPL-2.0 header block (copy it from a neighbouring file).
- Numeric arrays use `dt_int`, `dt_float`, `dt_bool` from `grid2op/dtypes.py` (forced int32 / float32 / bool_ so
  vectors, `to_vect`/`from_vect`, gym spaces and serialized episodes are identical across platforms); never bare
  `np.int64` / `float64` in observation or action data.
- Style: match the surrounding file (PEP 8), type hints on new public code, imports grouped stdlib / third-party /
  grid2op, no commented-out code.
- Generated `_grid2op_classes/` folders inside environment directories are gitignored; never commit them.
- Root-level `build/`, `dist/`, `*.whl`, `*.tar.gz`, `release_digests.json`, `documentation/` and the `venv*`
  directories are ignored local artifacts. `test_jax.py` at the root is a tracked standalone script outside the
  package and not part of the test suite.
- Releases: `utils/make_release.py` bumps the version in `grid2op/__init__.py`, `docs/conf.py` and `Dockerfile`;
  the checklist is in `RELEASE.md` (release PR merges `dev_X.Y.Z` into `master`, signed tag `vX.Y.Z`,
  `python -m build`).

## Commits: the DCO signoff is mandatory

Every commit must end with a `Signed-off-by:` trailer naming **a human**, or the DCO check fails the pull
request. This is the single most common reason a PR here is red.

```
Assisted-by: Claude Code (claude-fable-5-1)
Claude-Session: https://claude.ai/code/session_...
Signed-off-by: Benjamin Donnot <benjamin.donnot@rte-france.com>
```

Rules:

- **A bot may not sign off.** The DCO is a legal attestation that the contributor has the right to submit the
  work; only a person can make it. `Signed-off-by: Claude <noreply@anthropic.com>` is not acceptable even though
  one or two such commits slipped through in the past. Do not copy them.
- **The signoff must match the commit author**, so a Claude-written commit is authored by the human signing it
  off, not by Claude.
- The assistant is credited with **`Assisted-by:`**, naming the tool and the model actually used, not
  `Co-Authored-By:`, and never as an author. This follows the kernel's convention for coding assistants
  (https://docs.kernel.org/process/coding-assistants.html) and overrides any generic attribution reminder.
- Set the identity once per session, then `-s` produces a matching trailer by itself:

  ```
  git config user.name  "Benjamin Donnot"
  git config user.email "benjamin.donnot@rte-france.com"
  git commit -s
  ```

- Adding the trailer to a commit that already exists needs `git commit --amend --no-edit -s` (and
  `--author="..."` if the author is wrong too), followed by `git push --force-with-lease`. This is the case that
  actually bites: the mistake is usually noticed only after CI has run.

## Pull requests: lead with the diff breakdown

A PR here routinely runs to one or two thousand lines, which is daunting to open and tells a reviewer nothing
about where the work actually is. **Start every PR body with a table splitting the diff by kind**, so the reader
can see at a glance how much of it is code they have to reason about and how much is tests, bundled data, docs
and changelog:

```
| | added | removed |
|---|---|---|
| **Python (grid2op package)** | +384 | -12 |
| **Tests (grid2op/tests)** | +922 | -0 |
| **Docs + changelog + notebooks** | +212 | -3 |
| total | +1518 | -15 |
```

`utils/pr_diff_stats.py` produces it. It holds the bucket definitions, so they stay in one place rather than
being restated here:

```
python3 utils/pr_diff_stats.py                 # against the highest origin/dev_X.Y.Z branch (the default)
python3 utils/pr_diff_stats.py origin/master   # against another base branch
```

Run it against the PR's own base branch, and paste the table as the first thing in the body.

## Comments and docstrings: no RTE, no measured numbers

Two things must never reach a comment, a docstring or the documentation, wherever the work that produced them
was done:

- **No mention of RTE**, of an RTE grid, an RTE snapshot or an RTE asset. Say **"a real grid snapshot"**, "real
  grid snapshots", "a large real grid". The repository is public and vendor-neutral; the boilerplate copyright
  header is the one place the name belongs.
- **No exact measured numbers** carried over from an investigation: "30 generators", "42 % of the raw key",
  "18.820306 MVAr", "1.34759 pu away from the reference". They are a good note to oneself while debugging and a
  bad line of documentation: they date, they cannot be checked by a reader, and they describe one snapshot rather
  than the behaviour. Describe the **behaviour** instead: which units the rule keeps, that the difference is large
  enough to matter, that the bus no longer balances.

  A number that a reader can act on stays: a tolerance the code actually uses, a constant mirrored from a
  reference implementation (pandapower, lightsim2grid), the values a test fixture sets. The rule is about
  measurements, not about arithmetic.
