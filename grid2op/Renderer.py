import numpy as np
import cmath
import math # for regular real sqrt
import time
import pdb

try:
    import pygame
    import seaborn as sns
    can_plot = True
except:
    can_plot = False
    pass

__all__ = ["Renderer"]


class Point:
    # https://codereview.stackexchange.com/questions/70143/drawing-a-dashed-line-with-pygame
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))

    def __div__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))

    def __floordiv__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))

    def __truediv__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))

    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))

    # get back values in original tuple format
    def get(self):
        return (self.x, self.y)

    def to_cplx(self):
        return self.x + 1j * self.y

    @staticmethod
    def from_cplx(cplx):
        return Point((cplx.real, cplx.imag))


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    # https://codereview.stackexchange.com/questions/70143/drawing-a-dashed-line-with-pygame
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    for index in range(0, int(length/dash_length), 2):
        start = origin + (slope *    index    * dash_length)
        end   = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)


def draw_arrow(surf, color, start_pos, end_pos, positive_flow, width=1, num_arrows=10, length_arrow=15, angle_arrow=30):
    if positive_flow:
        origin = Point(start_pos)
        target = Point(end_pos)
    else:
        target = Point(start_pos)
        origin = Point(end_pos)

    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    # phi = cmath.phase(slope.to_cplx()) * 360 / 2*cmath.pi
    phi = cmath.phase(displacement.to_cplx()) * 360 / (2*cmath.pi)
    cste_ = 2*cmath.pi / 360 * 1j
    rotatedown = cmath.exp(cste_ * (180 + phi + angle_arrow) )
    rotateup = cmath.exp(cste_ * (180 + phi - angle_arrow) )

    first_arrow_part = length_arrow*rotateup
    second_arrow_part = length_arrow*rotatedown

    per_displ = displacement / (num_arrows+1)
    for index in range(0, int(num_arrows)):
        mid   = origin + (per_displ * (index + 1) )
        start_arrow = Point.from_cplx(mid.to_cplx() + first_arrow_part)
        end_arrow = Point.from_cplx(mid.to_cplx() + second_arrow_part)
        # , end_arrow.get()
        pygame.draw.lines(surf, color, False,
                            [start_arrow.get(), mid.get(), end_arrow.get()],
                            width)


class Renderer(object):
    """
    TODO
    """
    def __init__(self, substation_layout,
                 observation_space,
                 radius_sub=25.,
                 load_prod_dist=70.,
                 bus_radius=4.,
                 timestep_duration_seconds=1.,
                 fontsize=20):
        """

        Parameters
        ----------
        substation_layout: ``list``
            List of tupe given the position of each of the substation of the powergrid.

        observation_space: :class:`grid2op.Observation.ObservationHelper`
            Observation space

        """
        if not can_plot:
            raise RuntimeError("Impossible to plot as pygame cannot be imported.")

        self.observation_space = observation_space

        # pygame
        pygame.init()
        self.video_width, self.video_height = 1300, 700
        self.timestep_duration_seconds = timestep_duration_seconds
        self.screen = pygame.display.set_mode((self.video_width, self.video_height), pygame.RESIZABLE)
        pygame.display.set_caption('Grid2Op Renderer')  # Window title
        self.window_grid = (1000, 700)
        self.background_color = [70, 70, 73]
        self.screen.fill(self.background_color)
        self.lag_x = 150
        self.lag_y = 100
        self.font = pygame.font.Font(None, fontsize)
        self.font_pause = pygame.font.Font(None, 30)

        # graph layout
        # convert the layout that is given in standard mathematical orientation, to pygame representation (y axis
        # inverted)
        self._layout = {}
        tmp = [(el1, -el2) for el1, el2 in substation_layout]

        # then scale the grid to be on the window, with the proper margin (careful, margin are applied both left and r
        # and right, so count twice
        tmp_arr = np.array(tmp)
        min_ = tmp_arr.min(axis=0)
        max_ = tmp_arr.max(axis=0)
        b = min_
        a = max_ - min_
        tmp = [(int((el1- b[0]) / a[0] * (self.window_grid[0]  - 2*self.lag_x)) + self.lag_x,
                int((el2 - b[1]) / a[1] * (self.window_grid[1] - 2*self.lag_y)) + self.lag_y)
               for el1, el2 in tmp]

        self._layout["substations"] = tmp
        self.subs_elements = [None for _ in observation_space.sub_info]

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

    def _get_line_name(self, subor_id, sub_ex_id, line_id):
        l_nm = 'l_{}_{}_{}'.format(subor_id, sub_ex_id, line_id)
        return l_nm

    def _get_load_name(self, sub_id, c_id):
        c_nm = "load_{}_{}".format(sub_id, c_id)
        return c_nm

    def _get_gen_name(self, sub_id, g_id):
        p_nm = 'gen_{}_{}'.format(sub_id, g_id)
        return p_nm

    def event_looper(self, force=False):
        # TODO from https://github.com/MarvinLer/pypownet/blob/master/pypownet/environment.py
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
                if event.key == pygame.K_SPACE:
                    # pause_surface = self.draw_plot_pause()
                    # self.screen.blit(pause_surface, (320 + self.left_menu_shape[0], 320))
                    # pygame.display.flip()
                    return not force
        return force

    def render(self, obs, reward=None, done=None, timestamp=None):
        # TODO from https://github.com/MarvinLer/pypownet/blob/master/pypownet/environment.py
        force = self.event_looper(force=False)
        while self.event_looper(force=force):
            pass

        if not "line" in self._layout:
            # update the layout of the objects only once to ensure the same positionning is used
            # if more than 1 observation are displayed one after the other.
            self._compute_layout(obs)

        # The game is not paused anymore (or never has been), I can render the next surface
        self.screen.fill(self.background_color)

        # draw the state now
        self._draw_generic_info(reward, done, timestamp)
        self._draw_subs(observation=obs)
        self._draw_powerlines(observation=obs)
        self._draw_loads(observation=obs)
        self._draw_gens(observation=obs)

        pygame.display.flip()

    def _draw_generic_info(self, reward=None, done=None, timestamp=None):
        color = pygame.Color(255, 255, 255)
        if reward is not None:
            text_label = "Instantaneous reward: {:.1f}".format(reward)
            text_graphic = self.font.render(text_label, True, color)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 100))
        if done is not None:
            pass

        if timestamp is not None:
            text_label = "Date : {:%Y-%m-%d %H:%M}".format(timestamp)
            text_graphic = self.font.render(text_label, True, color)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 200))

    def _draw_subs(self, observation):
        for i, el in enumerate(self._layout["substations"]):
            self._draw_sub(el, radius=self.radius_sub)

    def _draw_sub(self, center, radius):
        pygame.draw.circle(self.screen,
                           pygame.Color(255, 255, 255),
                           [int(el) for el in center],
                           int(radius),
                           2)

    def _draw_powerlines(self, observation):

        for line_id, (rho, status, p_or) in enumerate(zip(observation.rho, observation.line_status, observation.p_or)):
            # the next 5 lines are always the same, for each observation, it makes sense to compute it once
            # and then reuse it

            sub_or_id, sub_ex_id = self._layout["line"][line_id]

            l_nm = self._get_line_name(sub_or_id, sub_ex_id, line_id)
            pos_or = self.subs_elements[sub_or_id][l_nm]["pos"]
            pos_ex = self.subs_elements[sub_ex_id][l_nm]["pos"]

            if not status:
                # line is disconnected
                draw_dashed_line(self.screen, pygame.Color(0, 0, 0), pos_or, pos_ex)
            else:
                # line is connected

                # step 0: compute thickness and color
                rho_max = 1.5  # TODO here get it back from parameters, or environment or whatever

                if rho < (rho_max / 1.5):
                    amount_green = 255 - int(255. * 1.5 * rho / rho_max)
                else:
                    amount_green = 0

                amount_red = int(255 - (50 + int(205. * rho / rho_max)))

                color = pygame.Color(amount_red, amount_green, 20)

                width = 1
                if rho > rho_max:
                    width = 4
                elif rho > 1.:
                    width = 3
                elif rho > 0.9:
                    width = 2
                width += 3

                # step 1: draw the powerline with right color and thickness
                pygame.draw.line(self.screen, color, pos_or, pos_ex, width)

                # step 2: draw arrows indicating current flows
                draw_arrow(self.screen, color, pos_or, pos_ex,
                           p_or >= 0.,
                           num_arrows=width,
                           width=width)

    def _get_position(self, theta):
        quarter_pi = cmath.pi / 4
        half_pi = cmath.pi / 2.

        if theta >= -quarter_pi and theta < quarter_pi:
            res = "center|left"
        elif theta >= quarter_pi and theta < quarter_pi + half_pi:
            res = "up|center"
        elif theta >= quarter_pi + half_pi and theta < quarter_pi + 2. * half_pi:
            res = "center|right"
        else:
            res = "down|center"

        return res

    def _aligned_text(self, pos, text_graphic, pos_text):
        pos_x = pos_text.real
        pos_y = pos_text.imag
        width = text_graphic.get_width()
        height = text_graphic.get_height()

        if pos == "center|left":
            pos_y -= height // 2
        elif pos == "up|center":
            pos_x -= width // 2
            pos_y -= height
        elif pos == "center|right":
            pos_x -= width
            pos_y -= height // 2
        elif pos == "down|center":
            pos_x -= width // 2
        self.screen.blit(text_graphic, (pos_x, pos_y))

    def _draw_loads(self, observation):
        for c_id, por in enumerate(observation.load_p):
            state = observation.state_of(load_id=c_id)
            sub_id = state["sub_id"]
            c_nm = self._get_load_name(sub_id, c_id)

            if not "elements_display" in self.subs_elements[sub_id][c_nm]:
                pos_load_sub = self.subs_elements[sub_id][c_nm]["pos"]
                pos_center_sub = self._layout["substations"][sub_id]

                z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
                theta = cmath.phase((self.subs_elements[sub_id][c_nm]["z"] - z_sub))
                pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

                # position of the end of the line connecting the object to the substation
                pos_end_line = pos_load - cmath.exp(1j * theta) * 20
                pos = self._get_position(theta)
                tmp_dict = {"pos_end_line": pos_end_line,
                            "pos_load_sub": pos_load_sub,
                            "pos_load": pos_load, "pos": pos}
                self.subs_elements[sub_id][c_nm]["elements_display"] = tmp_dict
            else:
                dict_element = self.subs_elements[sub_id][c_nm]["elements_display"]
                pos_end_line = dict_element["pos_end_line"]
                pos_load_sub = dict_element["pos_load_sub"]
                pos_load = dict_element["pos_load"]
                pos = dict_element["pos"]

            color = pygame.Color(0, 0, 0)
            width = 2
            pygame.draw.line(self.screen, color, pos_load_sub, (pos_end_line.real, pos_end_line.imag), width)
            text_label = "- {:.1f} MW".format(por)
            text_graphic = self.font.render(text_label, True, color)
            self._aligned_text(pos, text_graphic, pos_load)

    def _draw_gens(self, observation):
        for g_id, por in enumerate(observation.prod_p):
            state = observation.state_of(gen_id=g_id)
            sub_id = state["sub_id"]
            g_nm = self._get_gen_name(sub_id, g_id)

            if not "elements_display" in self.subs_elements[sub_id][g_nm]:
                pos_load_sub = self.subs_elements[sub_id][g_nm]["pos"]
                pos_center_sub = self._layout["substations"][sub_id]

                z_sub = (pos_center_sub[0] + 1j * pos_center_sub[1])
                theta = cmath.phase((self.subs_elements[sub_id][g_nm]["z"] - z_sub))
                pos_load = z_sub + cmath.exp(1j * theta) * self.load_prod_dist

                pos = self._get_position(theta)
                # position of the end of the line connecting the object to the substation
                pos_end_line = pos_load - cmath.exp(1j * theta) * 20
                tmp_dict = {"pos_end_line": pos_end_line, "pos_load_sub": pos_load_sub, "pos_load": pos_load,
                            "pos": pos}
                self.subs_elements[sub_id][g_nm]["elements_display"] = tmp_dict
            else:
                dict_element = self.subs_elements[sub_id][g_nm]["elements_display"]
                pos_end_line = dict_element["pos_end_line"]
                pos_load_sub = dict_element["pos_load_sub"]
                pos_load = dict_element["pos_load"]
                pos = dict_element["pos"]

            color = pygame.Color(0, 0, 0)
            width = 2
            pygame.draw.line(self.screen, color, pos_load_sub, (pos_end_line.real, pos_end_line.imag), width)
            text_label = "+ {:.1f} MW".format(por)
            text_graphic = self.font.render(text_label, True, color)
            self._aligned_text(pos, text_graphic, pos_load)

    def _draw_topos(self, observation, fig):
        #TODO copy paste from plotplotly
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
        #TODO copy paste from plotplotly
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



