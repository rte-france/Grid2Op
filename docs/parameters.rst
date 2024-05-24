.. _parameters-module:

Parameters
===================================
The challenge "learning to run a power network" offers different parameters to be customized, or to learn an
:class:`grid2op.Agent` that will perform better for example.

This class is an attempt to group them all inside one single structure.

For now, :class:`Parameters` have default value, but the can be read back / from json. Other serialization method will
come soon.

Example
--------

If you want to change the parameters it is better to do it at the creation of the environment.

This can be done with:

.. code-block:: python

    import grid2op
    from grid2op.Parameters import Parameters

    # Create parameters
    p = Parameters()
    
    # Disable lines disconnections due to overflows
    p.NO_OVERFLOW_DISCONNECTION = True
    
    # Allow 4 substations to be impacted each turn
    p.MAX_SUB_CHANGED = 4
    
    # Allow 10 lines actions per turn
    p.MAX_LINE_STATUS_CHANGED = 10

    # Give Parameters instance to make, so its used
    env = grid2op.make("l2rpn_case14_sandbox", param=p)


.. automodule:: grid2op.Parameters
    :members:
    :autosummary:

.. include:: final.rst
