Elements modeled in this environment and their main properties
===============================================================

Any grid2op environment model different elements. In this section, we explain what is modeled and what is not.

.. note:: Grid2Op do not assume any "power system" modeling. The backend is the only one responsible
    of maintaining the data it generates consistent from a power system point of view.

    The "modeling" here is to be understood as "data you can receive, and what they mean" and also time
    dependencies when it makes sense.

    It do not presume anything about any powersystem modeling.


The elements modeled are (work in progress):

- :ref:`generator`
- :ref:`load`
- :ref:`powerline`
- :ref:`shunt`
- :ref:`storage`
- :ref:`substation`

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

.. _generator:

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

You can modify the generator in different manner, from an **__action__** (NB some action do not allow the modification
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

In this section we explain the generators attributes you can access from an **__observation__**. These
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

Notations:
+++++++++++

Equations
++++++++++

.. _load:

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

.. _powerline:

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


.. _shunt:

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



.. _storage:

Storage units (optional)
------------------------


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


.. _substation:

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



