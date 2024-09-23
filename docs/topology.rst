

.. _topology-modeling-module:

Dive into the topology "modeling" in grid2op
===================================================================

In this page of the documentation we dive into the description of the 
"topology" of the grid in grid2op.

.. warning::
  Work in progress

.. note::
  You can also find another representation of the topology in grid2op
  in the page :ref:`detailed-topology-modeling-module` 


What do we call topology
---------------------------------

In the powersystem literature "topology" might refer to different things and
be encoded in different ways, for example there is the "nodal topology" which
is often use by the physical solvers (backends in case of grid2op), or there 
is the "detailed topoology" which rather uses swithes, breakers etc.

.. note::
  The "nodal topology" is a graph that meets the Kirchhoff Current Laws.

  The vertex of this graph are the "electrical node". These vertices contains, 
  from grid2op point of view, one or more "elements" of grid (side of powerline, 
  loads, generators, storage units etc.) that can be directly connected together.

  The edges of this graph are merging of 1 or more powerlines that connects two
  vertices together.

.. note::
  The "detailed topology" is more complicated. It also represents a graph but
  at a more granular level.

  In real powergrid, elements of the grid are connected together with switches /
  breakers / couplers etc. that can be either closed or opened.

  In real grid, the "topology" is controled with actions on these switches /
  breakers / couplers etc.

In the case of grid2op we adopt another representation for this "topology".
It is more detailed than containing purely the "nodal" information but
does not model the switches.

.. note::
  TODO have some illustrative examples here of "nodal" and "detailed"

  For example inspired from https://www.powsybl.org/pages/documentation/developer/tutorials/topology.html

.. note::
  This explanation is correct as of writing (September 2024) but there are
  some efforts to use a more detailed representation of the topology in 
  the form of `switches` in a branch in grid2op.

In plain English, the "topology" is a representation of the powergrid 
as a graph with the edges being the powerlines / transformers and the 
nodes being some "things" having attributes such that the power produced 
or consumed at this nodes.

As often in computer science, there are different ways to informatically 
represent a graph.

We chose to encode this "graph" in the form of a vector. This vector, 
often called the "topology vector" or "topo vect" has the following properties:

- it has as many component as the number of elements (load, generator, side of powerline
  or transformer, storage unit etc.) present in the grid. Each component of this vector
  provide information about the state of an unique element of the grid.
- it is a vector of integer (`=> -1`) with the following convention:
  
  - if a given component is `-1` this means the relevant element is connected
  - if a given component is `1` it means the element of the grid represented by this component is connected to "busbar 1"
  - if a given component is `2` it means the element of the grid is connected to "busbar 2"
  - etc. (for all `k >= 1` if a given component is `k` then it means the relevant element of the grid is connected to busbar `k`)
  - the component can never be `<= -2` nor `0`

This "topology vector" can change depending on the state of the grid.

Another "fixed" / "constant" / "immutable" information is needed to retrieve the
"topology" of the grid. It concerns the mapping between each elements of 
the grid and the "substation" to which it "connected".

.. note::
  The same word "connected" used here means two different things.

  The "connected to a substation" is independant of the status "connected / disconnected"
  of an element.

  Let's suppose the city of Nowhere is modeled by a load in the grid: 

  - "*Nowhere is connected to substation 5*" means that 
    the powergrid is made in such a way that the physical place where the transformer
    that powers the city of "Nowhere" is in a location that is called "substation 5".
    It can never be "disconnected" from substation 5 (this would mean the city ceased 
    to exist) nor can it be "connected to substation 1 [or 2, or 3, or 4, etc.]" 
    (this would mean this city magically change its geographical location and is 
    moved from a few hundred of miles / km)
  - "*Nowhere is disconnected*" means that the transformer
    powering the city of Nowhere is switched-off (blackout in this city)
  - "*Nowhere is connected to busbar 1*" means that
    within the "substation 5" there is an object called "busbar 1" and that 
    there is a "direct electrical path" (made of all closed switches) that
    connects the transformer of the city of Nowhere to this "busbar 1"

.. note::
  The mapping between each object and the substation to which it is connected
  does not change. This is why it is not stored in the topology vector.

This mapping is loadedonce and for all from the grid file by the "backend" at the 
creation of the environment. 

With both these information the "nodal topology" can be computed as followed:

- if an object is disconnected (associated component to the topology vector is `-1`)
  it is not connected (no kidding...) and can be ommitted when building the graph
- if two obejcts `o_i` and `o_j` are not "connected to the same substation" they 
  are not connected to the same vertex of the graph.
- if two objects `o_i` and `o_j` are "connected to the same substation" they are 
  part of the same "electrical node" (also called bus) if (and only if) 
  the associated component 
  of the "topoolgy vector" has the same integer. For example if the component 
  of the topology vector for `o_i` is 2 and the component for `o_j` is 1 
  they are NOT connected together. But if its component is 3 for `o_i`
  and 3 for `o_j` they are connected together.

.. note::
  As of writing, if a load or a generator is disconnected, there is a "game over".


Why the "switches" are not modled by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a grid modeling with switches, you can consult the dedicated 
page :ref:`detailed-topology-modeling-module` of the grid2op 
package.


.. warning::
  Doc in progress...

  
Switches are not in most synthetic grids
++++++++++++++++++++++++++++++++++++++++

There are no switches in most IEEE test cases which serve as reference 
for most of grid2op environment and are widely used in the literature. 
Forcing switches in grid2op would mean inventing them on these grid, which is
not necessary. When creating an open source environment, it would be 
mandatory to come up with a layout for each substation of the
fictive grid. And there are many different "substation layout" possible (
see *eg* https://www.technomaxme.com/understanding-busbar-systems/ )


Switches will not make the problem more realistic
+++++++++++++++++++++++++++++++++++++++++++++++++++

Switches information is too complicated to be manipulated correctly if we
consider time dependant states.Switches would also make the rules much more difficult to
implement. For example, in real time, some breakers can be opened / closed
while under charge but some other might not. This means an agent that would
operate the grid would have to anticipate to "pre configure" the switches 
"before" real time if it wants to adopt this and that. We believe that this 
is too complicated for an agent to do yet [TODO more info about that needed]

Closer to human reasoning
+++++++++++++++++++++++++++

As for our experience, human operators do not think in terms of opening / closing
switches. The first target a given "topology": these two elements connected together,
these other three also, but not with the previous ones etc. And then they 
use their expertise to find a combination of breakers which match what 
they want to achieve. We believe that the added value of AI is greater in the
first step (find the good nodal topology) so we decided to entirely skip the second 
one (which, we think, can be solved by optimization routines or heuristics)

Smaller action space
+++++++++++++++++++++

The problem we expose in grid2op is far from being solved (to our knowledge). And
we believe that making multiple consecutive small steps into the right direction is better than
modeling every bit of complexity of the "real" problem and then find a solution
to this really hard problem. Removing switches is a way to reduce the action space. Indeed,
if you consider the "grid2op standard" : "*maximum 2 independant buses per substation*" and
a substation with 4 elements. You need:

- an action space of **4 bits** with current grid2op modeling
  (one bit per elements)
- whereas you would need to "build" the substation layout, for example: 
  you create two busbars (one for each independant buses), then
  one switch connecting each of the 4 elements to both busbars plus possibly a 
  breaker between both busbars. Making **9 switches** / breakers in total.

.. note::
  Both type of action spaces would represent the same reality. This means that 
  in the second case lots of "possible action" would be ambiguous or lead finally
  to the "do nothing" action, which is not ideal.

In this case, adding switches would more than double (in this case) the size of the action space 
(4 btis without, 9 bits with them).

Simpler action and observaton spaces
+++++++++++++++++++++++++++++++++++++

One of the main issue with "topology" is that the same topology can be encoded differently.

With the proposed grid2op encoding this problem is not totally solved: the symmetry still exists. 
However it is drastically reduced from the symmetry there would have when manipulating directly
the switches.

Let's take again our example with a substation of 4 elements. For the "fully connected" topology,
the grid2op encoding can be either [1, 1, 1, 1] or [2, 2, 2, 2] which makes 2 solutions.

With the substation layout detailed in the paragraph `Smaller action space`_ it can be encoding with:

- [[1, 0], [1, 0], [1, 0], [1, 0], 0] : every element connected to busbar 1 and the busbar coupler between busbar 1 and 2 opened
- [[0, 1], [0, 1], [0, 1], [0, 1], 0] : every element connected to busbar 2 and the busbar coupler between busbar 1 and 2 opened
- [[1, 0], [1, 0], [1, 0], [1, 0], 1] : every element connected to busbar 1 and the busbar coupler between busbar 1 and 2 closed
- [[0, 1], [0, 1], [0, 1], [0, 1], 1] : every element connected to busbar 2 and the busbar coupler between busbar 1 and 2 closed
- [[1, 0], [0, 1], [0, 1], [0, 1], 1] : first element connected to busbar 1, all others to busbar 2 
  and the busbar coupler between busbar 1 and 2 closed
- [[0, 0], [1, 1], [0, 1], [0, 1], 1] : second element connected to busbar 1, all others to busbar 2 
  and the busbar coupler between busbar 1 and 2 closed
- ...

Basically, as long at the busbar coupler between busbar 1 and busbar 2 is closed, you can connect every element to every 
busbar and end-up with a valid encoding of the topology "fully connected".

In this representation, you have 2 + 2**4 = 18 possible "valid" encoding of the same "fully connected" topology.

.. note::
  We only count here "valid" topology, in the sense that an element is either connected to busbar 1 or busbar 2
  but not to both at the same time. But in fact it would be perfectly fine to connect and object to
  both busbar as long as the busbar coupler is closed (for each element this lead to 3 possible combination)

  There would be not 2**4 but 4**3 = 128 encoding of this "fully connected" topology.

  In general it is considered a good practice to chose a reprensentation that is as explicit and "unique"
  as possible.

Switches make the solver slightly slower
+++++++++++++++++++++++++++++++++++++++++

The switches information is also a reprensentation of the topology that is not the one used by the solver.

At some point, any solver will have to compute a (sparse) matrices and a (dense) vetor to represent
the physical laws. These are often computed by first reducing the "switches state" to the "nodal topology"
and then convert this graph to the proper matrix and vector.

By passing directly the "nodal topology" it is faster (for some solver at least) as the initial pre processing
of the switches state to the "graph" does not need to be performed.

.. note::
  And this why it is relatively hard for some "solver" to be used as a backend. 

  Some solver can only manipulate switches. In order to match grid2op representation, 
  it is then required to cast the "nodal topology" of grid2op to a switches state
  (which is for now HARD and slow), then pass these swtiches to the "solver".

  Afterwards, the "solver" will then run its internal routine (often really fast) 
  to retrieve the "nodal topology"
  of the grid (what the agent wanted to get) from the swtiches state.


It is easy to compute the grid2op representation from the switches
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is done internally by all solvers (pandapower when using switches but also all solver we 
know) at the initial state of running a powerflow and is relatively easy. Some graph 
alrogithms BFS (*eg* Breadth First Search) allows to quickly compute the "grid2op representation"
from the state of the switches.

This means that an agent can have full access to the switches, manipulate them and at the end
inform grid2op about the "grid2op topology" without too much trouble.

If we had modeled "by default" the switches it would mean that an agent that would "do like the human" 
(*ie* target a nodal topology) would then need to find some "switches states" that matches The
representation it targets. So an agent would have to do two things, instead of just one.

.. da,ger::
  To be honest, it also means that the current grid2op representation is not entirely "consistent".

  For some real grid, with some given substations layout, a agent could target a topology that is 
  not feasible: there does not exist a switches state that can represent this topology.

  This is currently a problem for real time grid operations. But we believe that a "routine" 
  (heuristic or optimization based) can be used to detect such cases.
  This routine is yet to be implemented (it is not on our priority list). The first step 
  (in our opinion) is to make a "proof of concept" that something can work. So basically
  that a  "target nodal topology" can be found.

  In a second stage, when things will be closer to production context, we will thing 
  about 

How it is accessible in grid2op
---------------------------------

.. warning::
  Doc in progress

The "topo_vect" vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
  Doc in progress

In the observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
  Doc in progress

In the action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
  Doc in progress

.. include:: final.rst
  