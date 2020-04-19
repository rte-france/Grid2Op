.. currentmodule:: grid2op.MakeEnv

Using pre defined Environments
===================================

Objectives
-----------
The function define in this module is the easiest and most convenient ways to create a valid
:class:`grid2op.Environment.Environment`.

To get started with such an environment, you can simply do:

..code-block:: python

    import grid2op
    env = grid2op.make()


You can consult the different notebooks in the `getting_stared` directory of this package for more information on
how to use it.

Created Environment should behave exactly like a gym environment. If you notice any unwanted behavior, please address
an issue in the official grid2op repository: `Grid2Op <https://github.com/rte-france/Grid2Op>`_

The environment created with this method should be fully compatible with the gym framework: if you are developing
a new algorithm of "Reinforcement Learning" and you used the openai gym framework to do so, you can port your code
in a few minutes (basically this consists in adapting the input and output dimension of your BaseAgent) and make it work
with a Grid2Op environment. An example of such modifications is exposed in the getting_started/ notebooks.

Important notes
---------------
As of version 0.7.1 a new function called ":func:`make_new`" has been developed in grid2op. This function, which will
replace the current implementation of :func:`make` in future versions, merges the behaviour of "grid2op.download"
script and "make" function.

It has the following behavior:

1) if you specify a full path to a local environment (containing the chronics and the default parameters),
   it will be used
2) if you specify the name of an environmnet that you have already downloaded, it will use this environment (NB
   currently no checks are implemented if the environment has been updated remotely, which can happen if
   we realize there were some issues with it.)
3) if the flag `local` is set to ``False`` (default behaviour) and none of the above conditions are met, the
   :func:`make_new` will download the data of this environment locally the first time it is called. If you don't want
   do download anything then you can pass the flag ``local=True``
4) if ``local=True`` (NON default behaviour) nothing will be loaded, and the :func:`make_new` will attempt to use a
   pre defined environment provided with the python package. We want to emphasize that because the environments provided
   with this package contains only little data, they are not suitable for leaning a consistent agent / controler. That
   is why a warning is sent in this case. Also, keep in mind that if you don't pass ``local=True`` then you will not
   have the possibility to search for these environments provided in the package.
5) if nothing is found, :func:`make_new` throws a EnvError.

TODO: explain behaviour of ".grid2opconfig.json"


Detailed Documentation by class
--------------------------------

.. automodule:: grid2op.MakeEnv
    :members:
    :autosummary:

.. include:: final.rst