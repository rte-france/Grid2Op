# What it does ?

In this example, we show explicitly the different steps performed when grid2op loads an "environment"
from the backend point of view.

It can be usefull for people wanting to implement a new backend for the grid2op platform.

Please refer to the documentation here https://grid2op.readthedocs.io/en/latest/createbackend.html 
for more information.

## Reminder

Grid2op is totally agnostic from the grid equations. In grid2op agents only manipulate "high level" objects
connected to a powergrid (for example "loads", "generators", "side of powerlines" etc.)

The way these objects behave and the equations they follow are totally irrelevant from the grid2op perspective. The task of making sure the proper equations are solved is carried out by the "backend".

Traditionnally, the "Backend" will rely on another tools that carries out the computation, implements the equaitons, solve it etc. In this setting, the "Backend" is some "glue code" that map the representation of your solver to grid2op expected functions. Some example of backend include:

- [PandapowerBackend](https://grid2op.readthedocs.io/en/latest/backend.html#grid2op.Backend.PandaPowerBackend): which is the default backend
- [EducPandaPowerBackend](https://github.com/rte-france/Grid2Op/blob/master/grid2op/Backend/EducPandaPowerBackend.py): which is a "simplification" of the previous backend for education purpose. So we highly recommend you to check it out :-)
- [lightsim2grid](https://lightsim2grid.readthedocs.io/en/latest/lightsimbackend.html#lightsim2grid.lightSimBackend.LightSimBackend) which is a backend that uses a port of some function of pandapower in c++ for speed.

We are also aware that some powerflows such as [Hades2](https://github.com/rte-france/hades2-distribution) and other commercial solvers such as PowerFactory are already connected with grid2op, so not open source at the moment.

Hopefully, more "powergrid solvers" can be connected.

## Note on static / dynamic, steady state / transient

At time of writing, only "steady state / static" solvers are connected to grid2op but that does not mean it cannot be different.

Grid2op only expects the "backend" to output some "state" about elements of the grid (for example "active flow at a given end of a powerline" or "voltage magnitude at the bus at which a generator is connected"). The way these "states" are computed is not important for grid2op nor for the backend. The only requirement is that these "states" can be accessed and retrieved python side.

## Alternative use

This example describe the "full" integration with grid2op directly. If your code can be linked against in c++ and you don't want to "worry" about grid2op "representation" / "encoding" / etc. (which is the topic of this example) you might directly want to compute the "complex voltage vector V" solution to "the equations". And in this case, if you can implement the function:

```cpp
bool compute_pf(const Eigen::SparseMatrix<cplx_type> & Ybus,  // the admittance matrix of your system
                CplxVect & V,  // store the results of the powerflow and the Vinit !
                const CplxVect & Sbus,  // the injection vector
                const Eigen::VectorXi & ref,  // bus id participating to the distributed slack
                const RealVect & slack_weights,  // slack weights for each bus
                const Eigen::VectorXi & pv,  // (might be ignored) index of the components of Sbus should be computed
                const Eigen::VectorXi & pq,  // (might be ignored) index of the components of |V| should be computed
                int max_iter,  // maximum number of iteration (might be ignored)
                real_type tol  // solver tolerance 
                );
```

then it should be relatively simple to use it with lightsim2grid.

**NB** this is not the preferred solution.

# Main "functions" to implement

We suppose that you already have a "solver" that is able to read a file describing a powergrid, retrieve the parameters needed, compute a solution to the equations and for which you can read the results. For example a "powerflow solver".

Once you have that, implementing a backend can be done in 4 different steps, each described in a subsection below.

## Step 1: loading the grid, exporting it grid2op side

This step is called only ONCE, when the grid2op environment is created. In this step, you read a grid file (in the format that you want) and the backend should inform grid2op about the "objects" on this powergrid and their location.

Basically, once you have loaded the file in your solver you should first fill `self.n_sub` (number of substations on your grid)

Then for each type of elements (among "load", "gen", "line_or", "line_ex" and "storage"), you fill :

- (optional) `self.name_$el` (*eg* self.name_load, self.name_line, self.name_storage) : the name of the elements (*eg* `self.name_load[5]` is the name of the load 5.). You can ignore it if you want and if that is the case, grid2op will automatically assign such names transparently.
- `self.n_$el` (*eg* self.n_load, self.n_gen, self.n_line, self.n_storage): the number of element. For example self.n_load will be the total number of loads on your grid.
- `self.$el_to_subid` (*eg* self.load_to_subid, self.gen_to_subid, self.line_or_to_subid, self.line_ex_to_subid, self.storage_to_subid): the id of the substation to which this given element is connected. For example `self.load_to_subid[2] = 5` informs grid2op that the load with id 2 is connected to substation with id 5 and `self.line_or_to_subid[7] = 9` informs grid2op that the origin side of line 7 is connected to substation with id 9.

You need to fill :
- integers: self.n_load, self.n_gen, self.n_storage, self.n_line, 
- vectos:  self.load_to_subid, self.gen_to_subid, self.line_or_to_subid, self.line_ex_to_subid

Then you call `self._compute_pos_big_topo()` and this will assign all the right vectors required by grid2op for you.

An example is given in the [Step1_loading](Step1_loading.py) script.

**NB** A "transformer" (from a powergrid perspective) is a "powerline" from a grid2op perspective.

## Step 2: modifying the state

## Step 3: solves the equations

## Step 4: reading the states
