.. _modeled-elements-module:

Elements modeled in this environment and their main properties
===============================================================

Any grid2op environment model different elements. In this section, we explain what is modeled and what is not.

.. note:: Grid2Op do not assume any "power system" modeling. The backend is the only one responsible
    of maintaining the data it generates consistent from a power system point of view.

    The "modeling" here is to be understood as "data you can receive, and what they mean" and also time
    dependencies when it makes sense.

    It do not presume anything about any powersystem modeling.


The elements modeled are (work in progress):

- :ref:`generator-mod-el`
- :ref:`load-mod-el`
- :ref:`powerline-mod-el`
- :ref:`shunt-mod-el`
- :ref:`storage-mod-el`
- :ref:`substation-mod-el`

Each type of elements will be described in the same way:

- `Description` will present a short description of what is the element
- `Static properties` is the subsection dedicated to the explanation about the properties of the
  given element that are static (same during all the episode) and acessible from all major grid2op
  class. For example "gen_pmin" is a static property of the "generators" and it can be accessed
  with `env.gen_pmin`, `env.action_space.gen_pmin`, `env.observation_space.gen_pmin`, `act.gen_pmin`
  or even `obs.gen_pmin` for example. We do not recommend to alter them.
- `Modifiable attributes` are the attributes than can be modified by the `action`
- `Observable attributes` are the attributs that can be read from the observation. We do not recommend
  to alter them.
- `Equations satisfied` explains the "constraint" of all of the above

.. _generator-mod-el:

Generators
-----------

Description
~~~~~~~~~~~~~~~~~~
Generators are elements connected to the powergrid who's role mainly consist in producing prower and
maintaining the safety of grid in some conditions (voltage collapse).

An a positive production means the generator produce something, so power is injected from the generator
to the grid.

Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

==========================   =============  =============================================================================================================================================================================================
Name                         Type            Description
==========================   =============  =============================================================================================================================================================================================
n_gen                         int           Total number of generators on the grid
name_gen                      vect, string  Names of all the generators
gen_to_subid                  vect, int     To which substation each generator is connected
gen_to_sub_pos                vect, int     Internal, see :ref:`create-backend-module`
gen_pos_topo_vect             vect, int     Internal, see :ref:`create-backend-module`
\* gen_type                   vect, string  Type of generator, among "nuclear", "hydro", "solar", "wind" or "thermal"
\* gen_pmin                   vect, float   Minimum production physically possible for each generator, in MW
\* gen_pmax                   vect, float   Maximum production physically possible for each generator, in MW
\* gen_redispatchable         vect, bool    For each generator, indicates if it can be "dispatched" see the subsection about the action for more information on dispatch
\* gen_max_ramp_up            vect, float   For each generator, indicates the maximum values the power can vary (upward) between two consecutive steps in MW. See the subsection about the equations for more information
\* gen_max_ramp_down          vect, float   For each generator, indicates the maximum values the power can vary (downward) between two consecutive steps in MW. See the subsection about the equations for more information
\* gen_min_uptime             vect, int     (currently unused) For each generator, indicates the minimum time a generator need to be "on" before being turned off.
\* gen_min_downtime           vect, int     (currently unused) For each generator, indicates the minimum time a generator need to be "off" before being turned on again.
\* gen_cost_per_MW            vect, float   (will change in the near future) Cost of production, in $ / MWh (in theory) but in $ / (MW . step) (each step "costs" `prod_p * gen_cost_per_MW`)
\* gen_startup_cost           vect, float   (currently unused) Cost to turn on each generator (in $)
\* gen_shutdown_cost          vect, float   (currently unused) Cost to turn off each generator (in $)
==========================   =============  =============================================================================================================================================================================================

(\* denotes optional properties available only for some environments)

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

You can modify the generator in different manner, from an **action** (NB some action do not allow the modification
of some of these attributes).

- `gen_set_bus`: set the bus to which the generator is connected. Usage: `act.gen_set_bus = [(gen_id, new_bus)]` where `gen_id` is the
  id of the generator you want to modify and `new_bus` the bus to which you want to connect it.
- `gen_change_bus`: change the bus to which the generator is connected. Usage: `act.gen_change_bus = gen_id` to change the bus of the
  generator with id `gen_id`.
- `redispatch`: will apply some redispatching a generator. Usage: `act.redispatch = [(gen_id, amount)]` to
  apply a redispatching action of `amount` MW on generator `gen_id`
- (internal) change the active production of a generator. Usage `act.update({"injection": {"prod_p": vect}}`
- (internal) change the voltage setpoint of a generator. Usage `act.update({"injection": {"prod_v": vect}}`

.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

In this section we explain the generators attributes you can access from an **observation**. These
attributes are:

- `gen_p`: the current active production of each generators, in MW. Usage: `obs.gen_p[gen_id]` will retrieve the
  active production of generator with id `gen_id`
- `gen_q`: the current reactive production of each generators, in MVAr. Usage: `obs.gen_q[gen_id]` will
  get the reactive production of generator with id `gen_id`
- `gen_v`: the voltage of the bus at which the generator is connected, in kV. Usage `obs.gen_v[gen_id]` will
  get the voltage magnitude of the bus at which generator with id `gen_id` is connected.
- `gen_bus`: the bus to which each generators is connected. Usage `obs.gen_bus[gen_id]` will
  get the bus to which generator with id `gen_id` is connected (typically -1, 1 or 2).
- `target_dispatch`: the target values given by the agent to the environment (*eg* using
  `act.redispatch`), in MW. Usage: `obs.target_dispatch[gen_id]`. More information in the "Equations" section.
- `actual_dispatch`: actual dispatch: the values the environment was able to provide as redispatching, in MW.
  Usage: `obs.actual_dispatch[gen_id]`. More information in the "Equations" section.

Satisfied equations
~~~~~~~~~~~~~~~~~~~~~~

Notations
+++++++++++
Let's denote by:

.. math::
    :nowrap:

    \begin{align*}
    \overline{\mathbf{g}} &: \text{maximum active production of all generators (vector)} \\
    \underline{\mathbf{g}} &: \text{minimum active production of all generators (vector)} \\
    \mathbf{\overline{\delta p}} &: \text{maximum ramp up for all generators}  \\
    \mathbf{\underline{\delta p}} &: \text{maximum ramp up for all generators} \\
    \mathbf{r}_t &: \text{vector of all redispatching asked by the agent at step t}  \\
    \mathbf{u}_t &: \text{vector of all active setpoint of all generators at step t}  \\
    \mathbf{h}_t &: \text{vector of all "target dispatch" at step t}  \\
    \mathbf{g}_t &: \text{vector of all active productions at step t} \\
    \mathbf{d}_t &: \text{vector of all actual redispatching at step t}  \\
    \end{align*}

Using the above notation, these vector are accessible in grid2op with:

- :math:`\overline{\mathbf{g}}` = `env.gen_pmax`
- :math:`\underline{\mathbf{g}}` = `env.gen_pmin`
- :math:`\mathbf{\overline{\delta p}}` = `env.gen_max_ramp_up`
- :math:`\mathbf{\underline{\delta p}}` = `env.gen_max_ramp_down`
- :math:`\mathbf{r}_t` = `act.redispatch`
- :math:`\mathbf{u}_t` = `act.prod_p` [typically read from the chronics]
- :math:`\mathbf{h}_t` = `obs.target_dispatch`
- :math:`\mathbf{g}_t` = `obs.prod_p`  [the production in the observation]
- :math:`\mathbf{d}_t` = `obs.actual_dispatch`

.. note:: Vector are denoted with bold font, like :math:`\mathbf{g}_t` and we will denote the ith component
    of this vector with :math:`g^i_t` (here representing then the active production of generator i at step t).
    We adopt the same naming convention for all the vectors.

    **NB** bold font might not work for some greek letters.

.. warning:: Unless told otherwise, the letters used here to write the equation are only relevant for the
    generators.

    It can happen the same letter is used multiple times for different element.

Equations
++++++++++
Generators have limit in the maximum / minimum power they can produce, this entails that:

.. math::
    :nowrap:
    :label: pmax

    \[\forall t, \underline{\mathbf{g}} \leq \mathbf{g}_t \leq \overline{\mathbf{g}}\]

Generators are also limited in the maximum / minimum varying power between consecutive steps, this
entails that:

.. math::
    :nowrap:
    :label: ramps

    \[\forall t , - \mathbf{\underline{\delta p}} \leq \mathbf{g}_{t+1} - \mathbf{g}_t \leq \mathbf{\overline{\delta p}}\]

The dispatch actions are cumulated in the "target_dispatch":

.. math::
    :nowrap:
    :label: targetdisp

    \[ \forall t,
        \left\{
        \begin{aligned}
            \mathbf{h}_{t+1} &= \mathbf{h}_t + \mathbf{r}_{t+1} \\
                             &= \sum_{v \leq t+1} \mathbf{r}_{v}
        \end{aligned}
        \right.
    \]

The total generation is the generation decided by the market (or a central authority) which
the agent modified with redispatching (for example because what the market / central authority decided
violate some security rules):

.. math::
    :nowrap:
    :label: updateg

    \[\forall t, \mathbf{g}_t = \mathbf{u}_t + \mathbf{d}_t\]

The redispatching is not supposed to impact the balancing between production and loads, which is supposed
to be ensured optimally (if the grid had an infinite capacity). This is why:

.. math::
    :nowrap:
    :label: zerosum

    \[\forall t, \sum_{\text{gen } i} d^i_t = 0\]

.. _gen_comp_redisp-mod-el:

Compute the redispatching vector
+++++++++++++++++++++++++++++++++

Because the agent do not know :math:`\mathbf{u}_t`, the redispatching action proposed by the agent
:math:`\mathbf{r}_{t}` is unlikely to meet equations :eq:`pmax`, :eq:`ramps`, :eq:`updateg` and
:eq:`zerosum`. This is why there is a difference between what is actually provided as redispatching
by the environment :math:`\mathbf{d}_{t}` and what the agent wanted to get :math:`\mathbf{r}_{t}`.

Currently, the way :math:`\mathbf{d}_{t}` is computed is by minimizing a distance
(based on the ramps) between the target dispatch "desired by the agent" :math:`\mathbf{h}_{t}` and
what is possible to get while satisfying the equations :eq:`pmax`, :eq:`ramps`, :eq:`updateg` and
:eq:`zerosum`. The routine to compute this 'actual dispatch' :math:`\mathbf{d}_{t}` uses the
"SLSQP" method of the `minimize` routine in the `scipy.optimize` module.

.. note:: Equation :eq:`zerosum` holds when they are no storage units on the grid. Please see the
    :ref:`storage-mod-el` section to get the "constraints" effectively implemented on the grid.

.. note:: The variable that can be modified by the optimisation routine are only the turned on dispatchable
    generators. The other generators (typically solar and wind) but also the storage units,
    are not modified when solving for this problem.

.. _load-mod-el:

Loads
-----------

Description
~~~~~~~~~~~~~~~~~~
TODO


Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

===========================  =============  =======================================
Name                          Type           Description
===========================  =============  =======================================
TODO
===========================  =============  =======================================

(\* denotes optional properties available only for some environments)

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

Equations satisfied
~~~~~~~~~~~~~~~~~~~~~~

TODO

.. _powerline-mod-el:

Powerlines
-----------

Description
~~~~~~~~~~~~~~~~~~
TODO


Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

===========================  =============  =======================================
Name                          Type           Description
===========================  =============  =======================================
TODO
===========================  =============  =======================================

(\* denotes optional properties available only for some environments)

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

Satisfied equations
~~~~~~~~~~~~~~~~~~~~~~

TODO


.. _shunt-mod-el:

Shunts (optional)
-----------------


Description
~~~~~~~~~~~~~~~~~~
TODO


Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

===========================  =============  =======================================
Name                          Type           Description
===========================  =============  =======================================
TODO
===========================  =============  =======================================

(\* denotes optional properties available only for some environments)

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

Satisfied equations
~~~~~~~~~~~~~~~~~~~~~~

TODO



.. _storage-mod-el:

Storage units (optional)
------------------------


Description
~~~~~~~~~~~~~~~~~~
Storage units are units that can act both as a production or a load. They have typically a certain
maximum energy the can store (when they are storing they take power from the grid to store it) that
can be discharge at any moment for a certain period (providing a certain maximum power for a given period
of time).

In grid2op, storage units have the `load` convention:

- a **positive** power means they are charging and thus **absorb** power from the grid (behaving like **load**)
- a **negative** power means they are discharging, and thus **inject** power to the grid (behaving like **generator**)

These storage units represents facilities that can store power in an industrial fashion. They are
typically pumped storage or batteries for example.

Some inspiration for the modeling of the storage units were provided by the NREL document:
https://www.greeningthegrid.org/news/new-resource-grid-scale-battery-storage-frequently-asked-questions-1

Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

===============================  =============  =======================================
Name                             Type           Description
===============================  =============  =======================================
n_storage                        int            Number of storage units on the grid
name_storage                     vect, str      Name of each storage units
storage_to_subid                 vect, int      Id of the substation to which each storage units is connected
storage_to_sub_pos               vect, int      Internal, see :ref:`create-backend-module`
storage_pos_topo_vect            vect, int      Internal, see :ref:`create-backend-module`
storage_type                     vect, str      Type of storage, among "battery" or "pumped_storage"
storage_Emax                     vect, float    For each storage unit, the maximum energy it can contains, in MWh
storage_Emin                     vect, float    For each storage unit, the minimum energy it can contains, in MWh
storage_max_p_prod               vect, float    For each storage unit, the maximum power it can give to the grid, in MW
storage_max_p_absorb             vect, float    For each storage unit, the maximum power it can take from the grid, in MW
storage_marginal_cost            vect, float    For each storage unit, the cost for taking / adding 1 MW to the grid, in $
storage_loss                     vect, float    For each storage unit, the self discharge, in MW, of the unit
storage_charging_efficiency      vect, float    For each storage unit, the "charging efficiency" (see bellow)
storage_discharging_efficiency   vect, float    For each storage unit, the "discharging efficiency" (see bellow)
===============================  =============  =======================================

(\* denotes optional properties available only for some environments)

The `storage_charging_efficiency` is a float between 0. and 1. If it's 1.0 it means that if the storage unit
absorb 1MW from the grid during 1h period, then 1MWh are added to the state of charge. If this efficiency is 0.5
then if 1MW is absorbed by the storage unit from the grid then only 0.5MWh will be stored in the unit.

It works symmetrically for `storage_discharging_efficiency`. For a storage unit, having a
`storage_discharging_efficiency` of 0.5 means that if the unit provide 1MW to the grid for 1h, then its
state of charge has been reduced by 2MWh (it would have been reduced by only 1MWh if this
efficiency was 1.0).

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

You can modify the generator in different manner, from an **action** (NB some action do not allow the modification
of some of these attributes).

- `storage_set_bus`: set the bus to which the storage unit is connected.
  Usage: `act.storage_set_bus = [(stor_id, new_bus)]` where `stor_id` is the
  id of the storage unit you want to modify and `new_bus` the bus to which you want to connect it.
- `storage_change_bus`: change the bus to which the storage unit is connected.
  Usage: `act.storage_change_bus = stor_id` to change the bus of the
  storage unit with id `stor_id`.
- `storage_p`: will tell the storage unit you want to get a given amount of power on the grid.
  Usage: `act.storage_p = [(stor_id, amount)]` to
  tell the storage unit `stor_id` to produce / absorb `amount` MW for the grid for the next step.


.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

In this section we explain the storage unit attributes you can access from an **observation**. These
attributes are:

- `storage_charge`: the state of charge of each storage unit, in MWh. Usage: `obs.storage_charge[sto_id]`
- `storage_power_target`: the power that was required from the last action of the agent, in MW
- `storage_power`: the power that is actually produced / absorbed by every storage unit.


Satisfied equations
~~~~~~~~~~~~~~~~~~~~~~

Notations
+++++++++++

Let's denote by:

.. math::
    :nowrap:

    \begin{align*}
        \Delta t & : \text{duration of a step (scalar  - usefull to get the energy from the power and vice versa)} \\
        \overline{\mathbf{E}} &: \text{maximum capacity of each of the storage units (vector)} \\
        \underline{\mathbf{E}} &: \text{maximum capacity of each of the storage units (vector)} \\
        \mathbf{\overline{p}} &: \text{maximum power that can be absorbed by the storage units (vector)}  \\
        \mathbf{\underline{p}} &: \text{maximum power that can be produced by the storage units (vector)} \\
        \mathbf{\overrightarrow{\rho}} &: \text{storage charging efficiency (vector)} \\
        \mathbf{\overleftarrow{\rho}} &: \text{storage discharging efficiency (vector)} \\
        \mathbf{l} &: \text{storage loss (vector)} \\
        \mathbf{u}_t &: \text{vector of all power consumption setpoint of all storage units at step t}  \\
        \mathbf{e}_t &: \text{vector representing the state of charge of the storage units at step t}  \\
        \mathbf{p}_t &: \text{vector of all actual consumption of all storage units at step t}  \\
    \end{align*}

Using the above notation, these vector are accessible in grid2op with:

- :math:`\overline{\mathbf{E}}` = `env.storage_Emax`
- :math:`\underline{\mathbf{E}}` = `env.storage_Emin`
- :math:`\mathbf{\overline{p}}` = `env.storage_max_p_absorb`
- :math:`\mathbf{\underline{p}}` = `env.storage_max_p_prod`
- :math:`\mathbf{\overrightarrow{\rho}}` = `env.storage_charging_efficiency`
- :math:`\mathbf{\overleftarrow{\rho}}` = `env.storage_discharging_efficiency`
- :math:`\mathbf{l}` = `env.storage_loss`
- :math:`\mathbf{u}_t` = `act.storage_p`  [the production / consumption setpoint, in the action]
- :math:`\mathbf{p}_t` = `obs.storage_power`  [the actual production / consumption, in the observation]
- :math:`\mathbf{e}_t` = `obs.storage_charge`

.. note:: Vector are denoted with bold font, like :math:`\mathbf{e}_t` and we will denote the ith component
    of this vector with :math:`e^i_t` (here representing then the active state of charge of
    storage unit i at step t).
    We adopt the same naming convention for all the vectors.

    **NB** bold font might not work for some greek letters.

.. warning:: Unless told otherwise, the letters used here to write the equation are only relevant for the
    generators.

    It can happen the same letter is used multiple times for different element.

Equations
++++++++++

In any case, the charge cannot be negative, and cannot be above the maximum (no there is not error here,
in some cases, the state of charge can appear to be slightly below the minimum, because of the losses):

.. math::
    :nowrap:
    :label: storagemax

    \[\forall t, 0 \leq \mathbf{e}_t \leq \overline{\mathbf{E}} \]

The storage charging / discharging equations are (keep in mind these are not the production / consumption
setpoint given in the action, but the production / setpoint available in the observation):

.. math::
    :nowrap:
    :label: charging

    \[ \forall \text{step } t, \forall \text{storage units } j,
        \left\{
        \begin{aligned}
            \text{if } p^j_t > 0, & e^j_t = e^j_t + \overrightarrow{\rho} . p^j_t . \Delta t & \text{ battery is charging} \\
            \text{if } p^j_t < 0, & e^j_t = e^j_t + \frac{1.0}{\overleftarrow{\rho}} . p^j_t . \Delta t & \text{ battery is discharging}
        \end{aligned}
        \right.
        \label{eq:charging}
    \]

There is a difference between the power setpoint and the actual implementation, mainly because there are
some constraint in the total amount of energy that can be stored in the unit. This translates into
a difference between the implemented storage production / consumption :math:`\mathbf{p}_t` and a the
setpoint in the action :math:`\mathbf{u}_t`:

.. math::
    :nowrap:
    :label: storageactual

    \[
        \begin{aligned}
            \min_{\mathbf{p}_t} & \left|\left| \mathbf{p}_t - \mathbf{u}_t \right|\right| \\
            \text{s.t.} & \\
                        & \text{if } p^j_t > 0, e^i_t + \overrightarrow{\rho} . p^j_t . \Delta t \leq \overline{\mathbf{E}}^i \\
                        & \text{if } p^j_t < 0, e^i_t + \frac{1.0}{\overleftarrow{\rho}} . p^j_t . \Delta t \geq \underline{\mathbf{E}}^i
        \end{aligned}
    \]


Currently this problem is not solved using an optimisation routine, but rather, if one of the constraints of
the :eq:`storageactual` is not met then the action is caped at the right value (*eg* if
:math:`e^j_t + \overrightarrow{\rho} . p^j_t > \overline{\mathbf{E}}^i` for one :math:`j` then
solving for :math:`p^j_t` the equation :math:`e^j_t + \overrightarrow{\rho} . p^j_t = \overline{\mathbf{E}}^j`)

As for the redispatching, the modification of the storage production / consumption
is not supposed to impact the balancing between production and loads, which is ensured by "the market"
(or a central authority). This means that, in case of presence of storage unit, the :eq:`zerosum`
showed in the :ref:`generator-mod-el` is modified as followed:

.. math::
    :nowrap:
    :label: storagemodif

    \[\forall t, \sum_{\text{gen } i} d^i_t + \sum_{\text{storage } j} p^j_t = 0\]

In the current implementation, this is done by substuting the equation :eq:`storagemodif` instead of
equation :eq:`zerosum` when solving
the optimization routine detailed in :ref:`gen_comp_redisp-mod-el`. The storage units are **NOT** modified
by this optimization routine.

Last, but not least, the storage loss is taken into account as followed:

.. math::
    :nowrap:
    :label: storageloss

    \[\forall t, \mathbf{e}_{t+1} = \mathbf{e}_{t} - \mathbf{l}.\Delta t \]

The equation :eq:`storageloss` supposes that :math:`\mathbf{e}_{t}` has been updated with the equations
:eq:`storagemax`, :eq:`charging`, :eq:`storageactual` and :eq:`storagemodif`.

.. note:: This is why, in the observation, you can get a "state of charge" (`obs.storage_charge`,
    :math:`\mathbf{e}_t`) below pmin because of the losses.

    If that is the case, even if no action is done by the agent, then some power will be taken
    from the grid to the storage unit to restore its capacity to the minimum capacity.

.. _substation-mod-el:

Substations
--------------

Description
~~~~~~~~~~~~~~~~~~
TODO


Static properties
~~~~~~~~~~~~~~~~~~
Their static properties are:

===========================  =============  =======================================
Name                          Type           Description
===========================  =============  =======================================
TODO
===========================  =============  =======================================

(\* denotes optional properties available only for some environments)

.. warning:: These attributes are static, and we do not recommend to alter them in any way. They are loaded at the
    start of the environment and should not be modified.

Modifiable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

.. note:: See the :ref:`action-module` and in particular the section
    :ref:`action-module-examples` for more information about how to manipulate these properties.

Observable attributes
~~~~~~~~~~~~~~~~~~~~~~

TODO

Satisfied equations
~~~~~~~~~~~~~~~~~~~~~~

TODO



