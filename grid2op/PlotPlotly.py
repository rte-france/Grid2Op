"""
This module provide a simple class to represent an :class:`grid2op.Observation.Observation` as a plotly graph.

We are aware that the graph can be largely improved. This tool is an example on what can be done with the Grid2Op
framework.

We hope It requires an valid installation of plotly and seaborn. These dependencies can be installed with:

.. code-block:: bash

    pip3 install grid2op[plots]

To use this plotting utilities, for example in a jupyter notebook please refer to the
``getting_started/4_StudyYourAgent`` notebook for more details information. The basic usage of this function is:

.. code-block:: python

    # make the relevant import
    from grid2op.MakeEnv import make
    from grid2op.PlotPlotly import PlotObs

    # create a simple toy environment
    environment = make("case5_example")

    # set up the plot utility
    graph_layout =  [(0,0), (0,400), (200,400), (400, 400), (400, 0)]
    plot_helper = PlotObs(substation_layout=graph_layout,
                          observation_space=environment.observation_space)

    # perform a step from this environment:
    do_nothing = environment.action_space({})
    environment.step(act)

    # do the actual plot
    fig = plot_helper.get_plot_observation(environment.get_obs())
    fig.show()

"""

import numpy as np
import cmath
import pdb

try:
    import plotly.graph_objects as go
    import seaborn as sns
    can_plot = True
except:
    can_plot = False
    pass

__all__ = ["PlotObs"]

# TODO add tests there


# Some utilities to plot substation, lines or get the color id for the colormap.
def draw_sub(pos, radius=50):
    """
    This function will draw the contour of a unique substation.

    Parameters
    ----------
    pos: ``tuple``
        It represents the position (x,y) of the center of the substation

    radius: ``float``
        Positive floating point representing the "radius" of the substation.

    Returns
    -------
    res: :class:`plotly.graph_objects.layout.Shape`
        A representation, as a plotly object of the substation

    """
    pos_x, pos_y = pos
    res = go.layout.Shape(
        type="circle",
        xref="x",
        yref="y",
        x0=pos_x - radius,
        y0=pos_y - radius,
        x1=pos_x + radius,
        y1=pos_y + radius,
        line_color="LightSeaGreen",
        fillcolor="lightgray"
    )
    return res


def get_col(rho):
    """
    Get the index (in the color palette) of the current  capacity usage.

    Parameters
    ----------
    rho: ``float``
        The capacity usage of a given powerline.

    Returns
    -------
    res: ``int``
        The integer (between 0 and 6) of this line capacity usage in terms of color.

    """
    if rho < 0.7:
        return 0
    if rho < 0.8:
        return 1
    if rho < 0.9:
        return 2
    if rho < 1.:
        return 3
    if rho < 1.1:
        return 5
    return 6


def draw_line(pos_sub_or, pos_sub_ex, rho, color_palette, status):
    """
    Draw a powerline with the color depending on its line capacity usage.

    Parameters
    ----------
    pos_sub_or: ``tuple``
        Position (x,y) of the origin end of the powerline

    pos_sub_ex: ``tuple``
        Position (x,y) of the extremity end of the powerline

    rho: ``float``
        Line capacity usage

    color_palette: ``object``
        The color palette to use

    status: ``bool``
        Powerline status (connected / disconnected). Disconnected powerlines are dashed.

    Returns
    -------
    res: :class:`plotly.graph_objects.layout.Shape`
        A representation, as a plotly object of the powerline

    """
    x_0, y_0 = pos_sub_or
    x_1, y_1 = pos_sub_ex

    res = go.layout.Shape(
        type="line",
        xref="x",
        yref="y",
        x0=x_0,
        y0=y_0,
        x1=x_1,
        y1=y_1,
        line=dict(
            color=color_palette[get_col(rho)] if status else "gray",  # 'cymk{}'.format(color_palette(rho))#color_palette(rho)
            dash=None if status else "dash"
        )
    )
    return res


class PlotObs(object):
    """
    This class aims at simplifying the representation of an observation as a plotly object given a layout of a given
    powergrid substation.
    It "automatically" handles the positionning of the powerlines, loads and generators based on that.

    This class is just here as an inspection tool. The results can be of course improved, epsecially the label of the
    powerlines, or the ppositioning of the loads and generators.

    Attributes
    ----------
    _layout: ``dict``
        Initial layout of the powergrid.

    subs_elements: ``list``
        For each substation, it gives a representation of all the object connected to it. So, for each substation, it
        has a dictionnary with:

            - key: the name of the objects
            - value: a dictionnary representing this object containing:

                - "type" : its type, among "load", "gen" and "line"
                - "sub_pos" (``int``) and index representing which element of the substation represents this object
                - "pos" : its position as a tuple
                - "z": its position as a complex number

    cols: ``object``
        A color palette, this should not be changed for now.

    radius_sub: ``float``
        The radius of each substation. The bigger this number, the better the topology will be visible, but the more
        space taken on the overall plot

    load_prod_dist: ``float``
        The distance between a load and a generator from the center of the substation. This must be higher than
        :attr:`PlotObs.radius_sub`

    bus_radius: ``float``
        The radius of the bus. When multiple buses are present in a substation, they are materialized by a filled
        circle. This number represents the size of these circles.



    """
    def __init__(self, substation_layout, observation_space,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.):
        """

        Parameters
        ----------
        substation_layout: ``list``
            List of tupe given the position of each of the substation of the powergrid.

        observation_space: :class:`grid2op.Observation.ObservationHelper`
            Observation space

        """
        if not can_plot:
            raise RuntimeError("Impossible to plot as plotly cannot be imported.")

        self._layout = {}
        self._layout["substations"] = substation_layout

        self.subs_elements = [None for _ in observation_space.sub_info]

        # define a color palette, whatever...
        sns.set()
        # pal = sns.dark_palette("palegreen")
        # pal = sns.color_palette("coolwarm", 7)
        # pal = sns.light_palette("red", 7)
        # self.cols = pal.as_hex()
        pal = sns.light_palette("darkred", 8)
        self.cols = pal.as_hex()[1:]

        self.radius_sub = radius_sub
        self.load_prod_dist = load_prod_dist # distance between load and generator to the center of the substation
        self.bus_radius = bus_radius
        # get the element in each substation
        for sub_id in range(observation_space.sub_info.shape[0]):
            this_sub = {}
            objs = observation_space.get_obj_connect_to(substation_id=sub_id)

            for c_id in objs["loads_id"]:
                c_nm = self._get_load_name(sub_id, c_id)
                this_load = {}
                this_load["type"] = "load"
                this_load["sub_pos"] = observation_space.load_to_sub_pos[c_id]
                this_sub[c_nm] = this_load

            for g_id in objs["generators_id"]:
                g_nm = self._get_gen_name(sub_id, g_id)
                this_gen = {}
                this_gen["type"] = "gen"
                this_gen["sub_pos"] = observation_space.gen_to_sub_pos[g_id]
                this_sub[g_nm] = this_gen

            for lor_id in objs["lines_or_id"]:
                ext_id = observation_space.line_ex_to_subid[lor_id]
                l_nm = self._get_line_name(sub_id, ext_id, lor_id)
                this_line = {}
                this_line["type"] = "line"
                this_line["sub_pos"] = observation_space.line_or_to_sub_pos[lor_id]
                this_sub[l_nm] = this_line

            for lex_id in objs["lines_ex_id"]:
                or_id = observation_space.line_or_to_subid[lex_id]
                l_nm = self._get_line_name(or_id, sub_id, lex_id)
                this_line = {}
                this_line["type"] = "line"
                this_line["sub_pos"] = observation_space.line_ex_to_sub_pos[lex_id]
                this_sub[l_nm] = this_line

            self.subs_elements[sub_id] = this_sub
        self.observation_space = observation_space

    def _get_line_name(self, subor_id, sub_ex_id, line_id):
        l_nm = 'l_{}_{}_{}'.format(subor_id, sub_ex_id, line_id)
        return l_nm

    def _get_load_name(self, sub_id, c_id):
        c_nm = "load_{}_{}".format(sub_id, c_id)
        return c_nm

    def _get_gen_name(self, sub_id, g_id):
        p_nm = 'gen_{}_{}'.format(sub_id, g_id)
        return p_nm

    def get_plot_observation(self, observation, fig=None):
        """
        Plot the given observation in the given figure.

        For now it represents information about load and generator active values.
        It also display dashed powerlines when they are disconnected and the color of each powerlines depends on
        its relative flow (its flow in amperes divided by its maximum capacity).

        If a substation counts only 1 bus, nothing specific is display. If it counts more, then buses are materialized
        by colored dot and lines will connect every object to its appropriate bus (with the proper color).

        Names of substation and objects are NOT displayed on this figure to lower the amount of information.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation to plot

        fig: :class:`plotly.graph_objects.Figure`
            The figure on which to plot the observation. Possibly ``None``, in this case a new figure is made.

        Returns
        -------
        res: :class:`plotly.graph_objects.Figure`
            The figure updated with the data from the new observation.
        """
        if fig is None:
            fig = go.Figure()

        # draw substation
        substation_layout = [draw_sub(el, radius=self.radius_sub) for i, el in enumerate(self._layout["substations"])]

        # draw name of substation
        # fig.add_trace(go.Scatter(x=[el for el, _ in self._layout["substations"]],
        #                          y=[el for _, el in self._layout["substations"]],
        #                          text=["sub_{}".format(i) for i, _ in enumerate(self._layout["substations"])],
        #                          mode="text",
        #                          showlegend=False))

        if not "line" in self._layout:
            # update the layout of the objects only once to ensure the same positionning is used
            # if more than 1 observation are displayed one after the other.
            self._compute_layout(observation)

        # draw powerlines
        lines = self._draw_powerlines(observation, fig)
        # draw the loads
        loads = self._draw_loads(observation, fig)
        # draw the generators
        gens = self._draw_gens(observation, fig)
        # draw the topologies
        topos = self._draw_topos(observation, fig)
        # update the figure with all these information
        fig.update_layout(shapes=substation_layout + lines + loads + gens + topos)

        # update legend, background color, size of the plot etc.
        fig.update_xaxes(range=[np.min([el for el, _ in self._layout["substations"]]) - 1.5 * (self.radius_sub + self.load_prod_dist),
                                np.max([el for el, _ in self._layout["substations"]]) + 1.5 * (self.radius_sub + self.load_prod_dist)],
                         zeroline=False)
        fig.update_yaxes(range=[np.min([el for _, el in self._layout["substations"]]) - 1.5 * (self.radius_sub + self.load_prod_dist),
                                np.max([el for _, el in self._layout["substations"]]) + 1.5 * (self.radius_sub + self.load_prod_dist)])
        fig.update_layout(
            margin=dict(
                l=20,
                r=20,
                b=100
            ),
            height=600,
            width=800,
            plot_bgcolor="white"
        )
        return fig

    def _draw_powerlines(self, observation, fig):
        """
        Draw the powerline, between two substations.

        Parameters
        ----------
        observation
        fig

        Returns
        -------

        """

        lines = []
        for line_id, (rho, status) in enumerate(zip(observation.rho, observation.line_status)):
            # the next 5 lines are always the same, for each observation, it makes sense to compute it once
            # and then reuse it

            state = observation.state_of(line_id=line_id)
            sub_or_id, sub_ex_id = self._layout["line"][line_id]

            l_nm = self._get_line_name(sub_or_id, sub_ex_id, line_id)
            pos_or = self.subs_elements[sub_or_id][l_nm]["pos"]
            pos_ex = self.subs_elements[sub_ex_id][l_nm]["pos"]

            # this depends on the grid
            # on this powergrid, thermal limit are not set at all. They are basically random.
            # so i multiply them by 300
            # rho *= 300
            lines.append(draw_line(pos_or, pos_ex,
                                   rho=rho,
                                   color_palette=self.cols,
                                   status=status))

            # TODO adjust position of labels...
            fig.add_trace(go.Scatter(x=[(pos_or[0] + pos_ex[0]) / 2],
                                     y=[(pos_or[1] + pos_ex[1]) / 2],
                                     text=["{:.1f}%".format(rho * 100)],
                                     mode="text",
                                     showlegend=False))
        return lines

    def _draw_loads(self, observation, fig):
        loads = []
        for c_id, por in enumerate(observation.load_p):
            state = observation.state_of(load_id=c_id)
            sub_id = state["sub_id"]
            c_nm = self._get_load_name(sub_id, c_id)

            pos_load_sub = self.subs_elements[sub_id][c_nm]["pos"]
            pos_center_sub = self._layout["substations"][sub_id]

            z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
            theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
            pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_load - cmath.exp(1j * theta) * 20

            # add the MW load
            fig.add_trace(go.Scatter(x=[pos_load.real],
                                     y=[pos_load.imag],
                                     text=["- {:.0f} MW".format(por)],
                                     mode="text",
                                     showlegend=False))
            # add the line between the MW display and the substation
            # TODO later one, add something that looks like a load, a house for example
            res = go.layout.Shape(
                type="line",
                xref="x",
                yref="y",
                x0=pos_end_line.real,
                y0=pos_end_line.imag,
                x1=pos_load_sub[0],
                y1=pos_load_sub[1],
                line=dict(
                )
            )
            loads.append(res)
        return loads

    def _draw_gens(self, observation, fig):
        gens = []
        for g_id, por in enumerate(observation.prod_p):
            state = observation.state_of(gen_id=g_id)
            sub_id = state["sub_id"]
            g_nm = self._get_gen_name(sub_id, g_id)

            pos_load_sub = self.subs_elements[sub_id][g_nm]["pos"]
            pos_center_sub = self._layout["substations"][sub_id]

            z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
            theta = cmath.phase((self.subs_elements[sub_id][g_nm]["z"] - z_sub))
            pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

            # position of the end of the line connecting the object to the substation
            pos_end_line = pos_load - cmath.exp(1j * theta) * 20

            # add the MW load
            fig.add_trace(go.Scatter(x=[pos_load.real],
                                     y=[pos_load.imag],
                                     text=["+ {:.0f} MW".format(por)],
                                     mode="text",
                                     showlegend=False))
            # add the line between the MW display and the substation
            # TODO later one, add something that looks like a load, a house for example
            res = go.layout.Shape(
                type="line",
                xref="x",
                yref="y",
                x0=pos_end_line.real,
                y0=pos_end_line.imag,
                x1=pos_load_sub[0],
                y1=pos_load_sub[1],
                line=dict(
                )
            )
            gens.append(res)
        return gens

    def _draw_topos(self, observation, fig):
        res_topo = []
        for sub_id, elements in enumerate(self.subs_elements):
            pos_center_sub = self._layout["substations"][sub_id]
            z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])

            tmp = observation.state_of(substation_id=sub_id)
            if tmp["nb_bus"] == 1:
                # not to overload the plot, if everything is at the same bus, i don't plot it
                continue
            # I have at least 2 buses

            # I compute the position of each elements
            bus_vect = tmp["topo_vect"]

            # i am not supposed to have more than 2 buses
            buses_z = [None, None]  # center of the different buses
            nb_co = [0, 0]  # center of the different buses

            # the position of a bus is for now the average of all the elements in there
            for el_nm, dict_el in elements.items():
                this_el_bus = bus_vect[dict_el["sub_pos"]] - 1
                if this_el_bus >= 0:
                    nb_co[this_el_bus] += 1
                    if buses_z[this_el_bus] is None:
                        buses_z[this_el_bus] = dict_el["z"]
                    else:
                        buses_z[this_el_bus] += dict_el["z"]
            buses_z = [el / nb for el, nb in zip(buses_z, nb_co)]
            theta_z = [cmath.phase((el - z_sub)) for el in buses_z]
            m_ = np.mean(theta_z) - cmath.pi / 2
            theta_z = [el-m_ for el in theta_z]
            buses_z = [z_sub + (self.radius_sub-self.bus_radius)*0.75*cmath.exp(1j * theta) for theta in theta_z]

            # TODO don't just do the average, but afterwards split it more evenly, and at a fixed distance from the
            # center of the substation

            # I plot the buses
            for bus_id, z_bus in enumerate(buses_z):
                bus_color = '#ff7f0e' if bus_id == 0 else '#1f77b4'
                res = go.layout.Shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=z_bus.real - self.bus_radius,
                    y0=z_bus.imag - self.bus_radius,
                    x1=z_bus.real + self.bus_radius,
                    y1=z_bus.imag + self.bus_radius,
                    fillcolor=bus_color,
                    line_color=bus_color,
                )
                res_topo.append(res)
            # i connect every element to the proper bus with the proper color
            for el_nm, dict_el in elements.items():
                this_el_bus = bus_vect[dict_el["sub_pos"]] -1
                if this_el_bus >= 0:
                    res = go.layout.Shape(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=dict_el["z"].real,
                        y0=dict_el["z"].imag,
                        x1=buses_z[this_el_bus].real,
                        y1=buses_z[this_el_bus].imag,
                        line=dict(color='#ff7f0e' if this_el_bus == 0 else '#1f77b4'))
                    res_topo.append(res)
        return res_topo

    def _compute_layout(self, observation):
        """
        Compute the position of each of the objects.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The observation used to know which object belong where.

        Returns
        -------

        """
        self._layout["line"] = {}

        # assign powerline coordinates
        for line_id in range(len(observation.rho)):
            if line_id not in self._layout["line"]:
                state = observation.state_of(line_id=line_id)
                sub_or_id = state["origin"]["sub_id"]
                sub_ex_id = state["extremity"]["sub_id"]
                pos_or = self._layout["substations"][sub_or_id]
                pos_ex = self._layout["substations"][sub_ex_id]

                # make sure the powerline are connected to the circle of the substation and not to the center of it
                z_or_tmp = pos_or[0] + 1j * pos_or[1]
                z_ex_tmp = pos_ex[0] + 1j * pos_ex[1]

                module_or = cmath.phase(z_ex_tmp - z_or_tmp)
                module_ex = cmath.phase(- (z_ex_tmp - z_or_tmp))

                # check parrallel lines:
                # for now it works only if there are 2 parrallel lines. The idea is to add / withdraw
                # 10° for each module in this case.
                # TODO draw line but not straight line in this case, this looks ugly for now :-/
                deg_parrallel = 25
                tmp_parrallel = self.observation_space.get_lines_id(from_=sub_or_id, to_=sub_ex_id)
                if len(tmp_parrallel) > 1:
                    if line_id == tmp_parrallel[0]:
                        module_or += deg_parrallel / 360 * 2 * cmath.pi
                        module_ex -= deg_parrallel / 360 * 2 * cmath.pi
                    else:
                        module_or -= deg_parrallel / 360 * 2 * cmath.pi
                        module_ex += deg_parrallel / 360 * 2 * cmath.pi

                z_or = z_or_tmp + self.radius_sub * cmath.exp(module_or * 1j)
                z_ex = z_ex_tmp + self.radius_sub * cmath.exp(module_ex * 1j)
                pos_or = z_or.real, z_or.imag
                pos_ex = z_ex.real, z_ex.imag
                self._layout["line"][line_id] = sub_or_id, sub_ex_id
                # TODO here get proper name
                l_nm = self._get_line_name(sub_or_id, sub_ex_id, line_id)

                self.subs_elements[sub_or_id][l_nm]["pos"] = pos_or
                self.subs_elements[sub_or_id][l_nm]["z"] = z_or
                self.subs_elements[sub_ex_id][l_nm]["pos"] = pos_ex
                self.subs_elements[sub_ex_id][l_nm]["z"] = z_ex

        # assign loads and generators coordinates
        # this is done by first computing the "optimal" placement if there were only substation (so splitting equally
        # the objects around the circle) and then remove the closest position that are taken by the powerlines.
        for sub_id, elements in enumerate(self.subs_elements):
            nb_el = len(elements)

            # equally split
            pos_sub = self._layout["substations"][sub_id]
            z_sub = pos_sub[0] + 1j * pos_sub[1]
            pos_possible = [self.radius_sub * cmath.exp(1j * 2 * cmath.pi * i / nb_el) + z_sub
                            for i in range(nb_el)]

            # remove powerlines (already assigned)
            for el_nm, dict_el in elements.items():
                if dict_el["type"] == "line":
                    z = dict_el["z"]
                    closest = np.argmin([abs(pos - z)**2 for pos in pos_possible])
                    pos_possible = [el for i, el in enumerate(pos_possible) if i != closest]

            i = 0
            # now assign load and generator
            for el_nm, dict_el in elements.items():
                if dict_el["type"] != "line":
                    dict_el["pos"] = (pos_possible[i].real, pos_possible[i].imag)
                    dict_el["z"] = pos_possible[i]
                    i += 1



