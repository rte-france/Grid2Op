
.. _troubleshoot_page:

Known issues and workarounds
===============================


In this section we will detail what are the common questions we have regarding grid2op and how to
best solve them (if we are aware of such a way...)

.. _troubleshoot_pickle:
    
Pickle issues
--------------------------

The most common (and oldest) issue regarding grid2op is its interaction with the `pickle` module
in python.

This module is used internally by the `multiprocessing` module and many others.

By default (and "by design") grid2op will create the classes when an environment 
is loaded. You can notice it like this:

.. code-block:: python

    import grid2op

    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name)

    print(type(env))

This will show something like `Environment_l2rpn_case14_sandbox`. This means that, 
not only the object `env` is created when you call `grid2op.make` but also
the class that `env` belongs too (in this case `Environment_l2rpn_case14_sandbox`).

.. note:: 
    We decided to adopt this design so that the powergrid reprensentation in grid2op
    is not copied and can be access pretty easily from pretty much every objects.

    For example you can call `env.n_gen`, `type(env).n_gen`, `env.backend.n_gen`,
    `type(env.backend).n_gen`, `obs.n_gen`, `type(obs).n_gen`, `act.n_gen`,
    `type(act).n_gen`, `env.observation_space.n_gen`, `type(env.observation_space).n_gen`
    well... you get the idea

    But allowing so makes it "hard" for python to understand how to transfer objects
    from one "process" to another or to save / restore it (indeed, python does not
    save the entire class definition it only saves the class names.)

This type of issue takes the form of an error with:

- `XXX_env_name` (*eg* `CompleteObservation_l2rpn_wcci_2022`) is not serializable.
- `_pickle.PicklingError`: Can't pickle <class 'abc._ObsEnv_l2rpn_case14_sandbox'>: attribute lookup _ObsEnv_l2rpn_case14_sandbox on abc failed

Automatic 'class_in_file'
+++++++++++++++++++++++++++

To solve this issue, we are starting from grid2op 1.10 to introduce some ways
to get around this automatically. It will be integrated incrementally to make
sure not to break any previous code.

The main idea is that grid2op will define the class as it used to (no change there)
but instead of keeping them "in memory" it will write it on the hard drive (in
a folder within the environment data) each time an environment is created.

This way, when pickle or multiprocessing will attempt to load the environment class,
they will be able to because the files are stored on the hard drive.

There are some drawbacks of course. The main one being that creating an environment 
can take a bit more time (especially if you have slow I/O). It will also use 
a bit of disk space (a few kB so nothing to worry about).

For now we tested it on multi processing and it gives promising results.

**TL;DR**: Enable this feature by calling `grid2op.make(env_name, class_in_file=True)` and you're good to go.

To enable this, you can:

- define a default behaviour by editing the `~/.grid2opconfig.json` global parameters
- define the environment variable `grid2op_class_in_file` **BEFORE** importing grid2op
- use the kwargs `class_in_file` when calling the `grid2op.make` function

.. note::
    In case of "conflicting" instruction grid2op will do the following:

    - if `class_in_file` is provided in the call to `grid2op.make(...)` it will use this and ignore everything else
    - (else) if the environment variable `grid2op_class_in_file` is defined, grid2op will use it
    - (else) if the configuration file is present and the key `class_in_file` is there, grid2op will 
      use it
    - (else) it will use its default behaviour (as of writing, grid2op 1.10.3) it is to **DEACTIVATE**
      this feature (in the near future the default will change and it will be activated by default)

For example:

The file `~/.grid2opconfig.json` can look like:

.. code-block:: json

    {
        "class_in_file" : false
    }

or 
.. code-block:: json

    {
        "class_in_file" : true
    }

If you prefer to work with environment variables, we recommend you do something like :

.. code-block:: python

    import os

    os.environ["grid2op_class_in_file"] = "true"  # or "false" if you want to disable it

    import grid2op

And if you prefer to use it directly in `grid2op.make(...)` funciton, you can do it with:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    env = grid2op.make(env_name, class_in_file=True) # or `class_in_file=False`


If you want to know if you environment has used this new feature, you can check with:

.. code-block:: python

    import grid2op
    env = grid2op.make(...)
    print(env.classes_are_in_files())

.. danger::
    If you use this, make sure (for now) that the original grid2op environment that you have created
    is not deleted. If that is the case then the folder containing the classes definition will be
    removed and you might not be able to work with grid2op correctly.


Experimental `read_from_local_dir`
+++++++++++++++++++++++++++++++++++

Before grid2op 1.10.3 the only way to get around pickle / multiprocessing issue was a "two stage" process:
you had first to tell grid2op to generate the classes and then to tell it to use it in all future environment.

This had the drawbacks that if you changed the backend classes, or the observation classes or the
action classes, you needed to start the whole process again. ANd it as manual so you might have ended up
doing some un intended actions which could create some "silent bugs" (the worst kind, like for example 
not using the right class...)

To do it you first needed to call, once (as long as you did not change backend class or observation or action etc.) 
in a **SEPARATE** python script:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"  # or any other name

    env = grid2op.make(env_name, ...)  # again: redo this step each time you customize "..."
    # for example if you change the `action_class` or the `backend` etc.

    env.generate_classes()


And then, in another script, the main one you want to use:

.. code-block:: python

    import grid2op
    env_name = SAME NAME AS ABOVE
    env = grid2op.make(env_name,
                        experimental_read_from_local_dir=True,
                        SAME ENV CUSTOMIZATION AS ABOVE)

As of grid2op 1.10.3 this process can be made automatically (not without some drawbacks, see above). It might
interact in a weird (and unpredictable) way with the `class_in_file` so we would recommend to use one **OR**
(exclusive OR, XOR for the mathematicians) the other but avoid mixing the two:

- either use `grid2op.make(..., class_in_file=True)`
- or use `grid2op.make(..., experimental_read_from_local_dir=True)`



.. include:: final.rst
