# What it does ?

In this example, we show explicitly the different steps performed when grid2op loads an "environment"
from the backend point of view.

It can be usefull for people wanting to implement a new backend for the grid2op platform.

Please refer to the documentation here https://grid2op.readthedocs.io/en/latest/createbackend.html 
for more information.

Basically, the typical "grid2op use" is:

```python

# called once
backend.load_grid(...)

# called for each "step", thousands of times
backend.apply_action()  # modify the topology, load, generation etc.
backend.runpf()  # run the solver

backend.get_topo_vect() # retrieve the results
backend.loads_info() # retrieve the results
backend.generators_info() # retrieve the results
backend.lines_or_info() # retrieve the results
backend.lines_ex_info() # retrieve the results
```

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

This is done by the method:

```python
def load_grid(self, path, filename=None):
    TODO !
```

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

This step is "first step" of the "grid2Op backend loop" (which is summarized by: "modify", "run the model", "retrieve the state of the elemtns", repeat).

It is implemented in the method `apply_action` (that does not return anything):

```python
def apply_action(self, action=None):
    TODO !
```

Classically, you can divide this method into different modifications:
- continuous modifications: change the active / reactive consumption of loads or storage units, the active power at generators or the voltage setpoint at these generators.
- discrete / topological modifications: connect / disconnect powerlines or change the bus to which an element is connected.

To implement it, you simply need to implement all the above part. Detailed examples are provided in the scripts "StepK_change_load.py" or "StepK_change_gen.py" for examples. Indeed we not find convenient to test "simply" that the setpoint has been modified. We prefer testing that the setpoint can be changed and then that the results can be read back (see steps 3 and 4 below).

**NB** the "action" here is NOT a grid2op.Action.BaseAction. It is a grid2op.Action._BackendAction !

## Step 3: solves the equations

This is the second step of the "grid2op backend loop" (which is still "modify", "run", "retrieve the results"). It is implemented in the function:

```python
def runpf(self, is_dc: bool=False) -> Tuple[bool, Union[None, Exception]]:
    TODO
    return has_converged, exception_if_diverged_otherwise_None
```

This is probably the most straightforward function to implement as you only need to call something like 'compute()' or 'run_pf' or 'solve' on your underlying model.

Detailed examples are provided in the scripts "StepK_change_load.py" or "StepK_change_gen.py" for examples. Indeed we not find convenient to test "simply" that the setpoint has been modified. We prefer testing that the setpoint can be changed and then that the results can be read back (see step 4 below).

## Step 4: reading the states

This is the third and final "call" of the "grid2op backend loop". At this stage, you are expected to export the results of your computation python side. Results should follow some given convention (*eg* units).

It is implemented in the functions:

```python
    def get_topo_vect(self):
        TODO
    
    def loads_info(self):
        TODO

    def generators_info(self):
        TODO
    
    def lines_or_info(self):
        TODO
    
    def lines_ex_info(self):
        TODO
```

Detailed examples are provided in the scripts "StepK_change_load.py" or "StepK_change_gen.py" for examples where the whole "backend loop" is exposed "element by element".

More explicitely:

- **get_topo_vect(self):** returns the topology vector
- **generators_info(self):** returns gen_p (in MW), gen_q (in MVAr), gen_v (in kV) [gen_v is the voltage magnitude at the bus to which the generator is connected]
- **loads_info(self):** returns load_p (in MW), load_q (in MVAr), load_v (in kV) [load_v is the voltage magnitude at the bus to which the generator is connected]
- **lines_or_info(self):** returns p_or (in MW), q_or (in MVAr), v_or (in kV), a_or (in A) [all the flows at the origin side of the powerline (remember it includes "trafo") + the voltage magnitude at the bus to which the origin side of the powerline is connected]
- **lines_or_info(self):** returns p_ex (in MW), q_ex (in MVAr), v_ex (in kV), a_ex (in A) [all the flows at the extremity side of the powerline (remember it includes "trafo") + the voltage magnitude at the bus to which the extremity side of the powerline is connected]


## Breakpoint :-)

At this stage, you can already use your backend with grid2op and all its eco system, even though some functionalities might still be missing (seed the "advanced" features below). The scripts `Step0` to `Step6` propose a possible way to split the coding
of all these functions into different independant tasks and to have basic "tests" (more preciselys examples of what could be some tests).

More precisely:

- [Step0_make_env](./Step0_make_env.py): create a grid2op environment that you can use even if your backend is not completely coded.
  It does that by relying on the computation of the powerflow by the default backend (of course it assumes the same grid can be loaded 
  by your backend and by Pandapower, which might involve converting some data from pandapower format (json specific representation)
  to your format. You can use some utilities of pandapower for such purpose, see *eg* https://pandapower.readthedocs.io/en/latest/converter.html)
- [Step1_loading](./Step1_loading.py): gives and example on how to implement the "load the grid from a file and define everything needed by grid2op"
- [Step2_modify_load](./Step2_modify_load.py): gives and example on how to implement the powerflow and on how to modify the load setpoints
- [Step3_modify_gen](./Step3_modify_gen.py): gives and example on how to modify the generator setpoints
- [Step4_modify_line_status](./Step4_modify_line_status.py): gives and example on how to modify powerline status (disconnect / reconnect) powerlines
- [Step5_modify_topology](./Step5_modify_topology.py): gives some examples on the topology changes connect object to different busbars
  at the substation they are connected to.
- [Step6_integration](./Step6_integration.py): gives some examples of agents interacting on the grid (and powerflow are carried out by your backend)


## (advanced): automatic testing

TODO

How to use grid2op tests to test your backend in depth?

## (advanced): handling of storage units

TODO (do not forget the storage description file !)

## (advanced): handling of shunts

TODO

## (advanced): handling of other generators attributes (cost, ramps, pmin / pmax etc.)

TODO

## (advanced): copy

TODO