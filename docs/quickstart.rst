Getting started
===================================

In this chapter we present how to install grid2op.

############
Installation
############

*************
Requirements
*************
This software uses python (at tested with version >= 3.6).

To install it i's also recommended to have `git`.

*************
Installation
*************

Using pip (recommended)
++++++++++++++++++++++++
Grid2op is hosted on pypi and can be installed like most python package with:

.. code-block:: bash

    pip install grid2op

It should be now installed. Don't hesitate to visit the section `Start Using grid2op`_ for more information on its
usage or the :ref:`grid2op-module` for a more in depth presentation of this package. If you
would rather start directly to interact with a powergrid you can visit the :ref:`make-env-module`.

From source (advanced user)
+++++++++++++++++++++++++++
If you want to develop new grid2op module (for example a new types of Backend, or a new kind of Chronics to
read new types of data) this section is made for you.


First, it is recommended (but optionnal) to make a virtual environment:

.. code-block:: bash

    pip3 install -U virtualenv

The second step is to clone the Grid2Op package (`git` is required):

.. code-block:: bash

    git clone https://github.com/rte-france/Grid2Op.git
    cd Grid2Op
    python3 -m virtualenv venv_grid2op

This should create a folder Grid2Op with the current sources.

Then the installation script of Grid2Op can be run to install the current simulator
(including the Python libraries dependencies):

.. code-block:: bash

    cd Grid2Op/
    source venv_grid2op/bin/activate
    pip install -U .


After this, this simulator is available under the name grid2op (from a python console)

.. code-block:: python

    import grid2op

####################
Start Using grid2op
####################
To get started into the grid2op ecosystem, we made a set of notebooks
that are available, without any installation thanks to
`Binder <https://mybinder.org/v2/gh/rte-france/Grid2Op/master>`_ . Feel free to visit the "getting_started" page for
more information and a detailed tour about the issue that grid2op tries to address.