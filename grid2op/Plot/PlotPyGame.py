# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This module defines the :class:`Renderer` that is able to display the state (:class:`grid2op.BaseObservation.BaseObservation`)
of the powergrid on a dedicated window.

It is also able to output a 3d representation of this representation to be further used by other libraries to
output gifs for example.

"""

import numpy as np
import cmath
import math
import os
import time

from grid2op.Plot.BasePlot import BasePlot
from grid2op.Exceptions.PlotExceptions import PyGameQuit, PlotError

try:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    import pygame
    can_plot = True
except Exception as e:
    can_plot = False
    pass


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


def _draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
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


def _draw_arrow(surf, color, start_pos, end_pos, positive_flow, width=1, num_arrows=10,
                length_arrow=10, angle_arrow=30):
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


class PlotPyGame(BasePlot):
    """

    .. warning:: /!\\\\ This class is deprecated /!\\\\

    This renderer should be used only for "online" representation of a powergrid.

    """
    def __init__(self,
                 observation_space,
                 substation_layout=None,
                 radius_sub=20.,
                 load_prod_dist=70.,
                 bus_radius=5.,
                 timestep_duration_seconds=1.,
                 fontsize=20):
        """

        Parameters
        ----------
        substation_layout: ``list``
            List of tupe given the position of each of the substation of the powergrid.

        observation_space: :class:`grid2op.Observation.ObservationSpace`
            BaseObservation space used for the display

        radius_sub: ``int``
            radius (in pixel) of the substations representation.

        load_prod_dist: ``int``
            distance (in pixels) between the substation and the load or the generator.

        bus_radius: ``int``
            The buses are represented by small circles. This is the radius (in pixel) for the pixels representing
            the buses.

        timestep_duration_seconds: ``float``
            Minimum time during which a time step will stay on the screen, in second.

        fontsize: ``int``
            size of the font used to display the texts.


        """
        if not can_plot:
            raise PlotError("Impossible to plot as pygame cannot be imported.")

        self.window_grid = (1000, 700)
        self.lag_x = 150
        self.lag_y = 100

        BasePlot.__init__(self,
                          substation_layout=substation_layout,
                          observation_space=observation_space,
                          radius_sub=radius_sub,
                          load_prod_dist=load_prod_dist,
                          bus_radius=bus_radius)

        # pygame
        self.__is_init = False
        self.video_width, self.video_height = 1300, 700
        self.timestep_duration_seconds = timestep_duration_seconds
        self.time_last = None
        self.fontsize = fontsize
        self.background_color = [70, 70, 73]

        # init pygame
        self.display_called = None
        self.screen = None
        self.font = None
        self.init_pygame()

        # pause button
        self.font_pause = pygame.font.Font(None, 30)
        self.color_text = pygame.Color(255, 255, 255)
        self.text_paused = self.font_pause.render("Game Paused", True, self.color_text)

        # maximum overflow possible
        self.rho_max = 2.

        # utilities
        self.cum_reward = 0.
        self.nb_timestep = 0

        # colors
        self.col_line = pygame.Color(0, 0, 255)
        self.col_sub = pygame.Color(255, 0, 0)
        self.col_load = pygame.Color(0, 0, 0)
        self.col_gen = pygame.Color(0, 255, 0)
        self.default_color = pygame.Color(0, 0, 0)

        # deactivate the display on the screen
        self._deactivate_display = False

    def change_duration_timestep_display(self, new_timestep_duration_seconds):
        """
         .. warning:: /!\\\\ This class is deprecated /!\\\\

        Change the duration on which the screen is displayed.
        """
        self.timestep_duration_seconds = new_timestep_duration_seconds

    def init_pygame(self):
        if self.__is_init is False:
            pygame.init()
            self.display_called = False
            self.screen = pygame.display.set_mode((self.video_width, self.video_height), pygame.RESIZABLE)
            self.font = pygame.font.Font(None, self.fontsize)
            self.__is_init = True

    def reset(self, env):
        """
         .. warning:: /!\\\\ This class is deprecated /!\\\\

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The used environment.

        Returns
        -------

        """
        self.cum_reward = 0.
        self.nb_timestep = 0
        self.rho_max = env.parameters.HARD_OVERFLOW_THRESHOLD

    def _get_sub_layout(self, init_layout):
        tmp = [(el1, -el2) for el1, el2 in init_layout]

        # then scale the grid to be on the window, with the proper margin (careful, margin are applied both left and r
        # and right, so count twice
        tmp_arr = np.array(tmp)
        min_ = tmp_arr.min(axis=0)
        max_ = tmp_arr.max(axis=0)
        b = min_
        a = max_ - min_
        res = [(int((el1- b[0]) / a[0] * (self.window_grid[0]  - 2*self.lag_x)) + self.lag_x,
                int((el2 - b[1]) / a[1] * (self.window_grid[1] - 2*self.lag_y)) + self.lag_y)
               for el1, el2 in tmp]
        return res

    def _event_looper(self, force=False):
        has_quit = False
        if self._deactivate_display:
            return force, has_quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                has_quit = True
                return force, has_quit
                # pygame.quit()
                # exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    has_quit = True
                    return force, has_quit
                if event.key == pygame.K_SPACE:
                    self._get_plot_pause()
                    # pause_surface = self.draw_plot_pause()
                    # self.screen.blit(pause_surface, (320 + self.left_menu_shape[0], 320))
                    return not force, has_quit
        return force, has_quit

    def _press_key_to_quit(self):
        """

         .. warning:: /!\\\\ This class is deprecated /!\\\\

        This utility function waits for the player to press a key to exit the renderer (called when the episode is done)

        Returns
        -------
        res: ``bool``, ``bool``
            ``True`` if the human player closed the window, in this case it will stop the computation: no other episode
            will be computed. ``False`` otherwise.

        """
        if self._deactivate_display:
            return
        has_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                has_quit = True
                return True, has_quit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    has_quit = True
                    return True, has_quit
                if event.key == pygame.K_SPACE:
                    return True, has_quit
        return False, has_quit

    def close(self):
        """
        This method is called when the renderer should be close.
        """
        self.display_called = False
        try:
            self._quit_and_close()
        except PyGameQuit:
            pass

    def _get_plot_pause(self):
        position = 300
        start_pause = position + self.text_paused.get_height()
        end_pause = start_pause + 50
        y_text_left = self.window_grid[0] + 100
        self.screen.blit(self.text_paused, (y_text_left, 300))

        pygame.draw.line(self.screen,
                         self.color_text,
                         (y_text_left+self.text_paused.get_width()//2-10, start_pause),
                         (y_text_left+self.text_paused.get_width()//2-10, end_pause),
                         10)
        pygame.draw.line(self.screen,
                         self.color_text,
                         (y_text_left+self.text_paused.get_width()//2+10, start_pause),
                         (y_text_left + self.text_paused.get_width()//2+10, end_pause),
                         10)
        pygame.display.flip()

    def _draw_final_information(self, reward, done, timestamp):
        if done is not None:
            if done:
                text_label = "GAME OVER, press any key to continue to next episode."
                text_graphic = self.font.render(text_label, True, self.color_text)
                self.screen.blit(text_graphic, (self.window_grid[0]+100, 100))
                text_label = "Total cumulated reward: {:.1f}".format(self.cum_reward)
                text_graphic = self.font.render(text_label, True, self.color_text)
                self.screen.blit(text_graphic, (self.window_grid[0]+100, 130))
                text_label = "Total number timesteps: {:.1f}".format(self.nb_timestep)
                text_graphic = self.font.render(text_label, True, self.color_text)
                self.screen.blit(text_graphic, (self.window_grid[0]+100, 160))

    def _quit_and_close(self):
        # pygame.reset_vars()
        # pygame.gameLoop()
        pygame.display.quit()
        # pygame.quit()
        self.display_called = None
        self.screen = None
        self.font = None
        self.__is_init = False
        raise PyGameQuit()

    def deactivate_display(self):
        self._deactivate_display = True

    def get_rgb(self, obs, reward=None, done=None, timestamp=None):
        """

         .. warning:: /!\\\\ This class is deprecated /!\\\\

        Computes and returns the rgb 3d array from an observation, and potentially other informations.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The observation to converte into a 3d array

        reward: ``float``
            The current reward

        done: ``bool``
            Whether this is the last frame of the episode.

        timestamp: ``datetime.datetime``
            The curent datetime corresponding to the observation

        Returns
        -------
        res: ``numpy.ndarray``
            The 3d representation of the observation that can then be converted to a gif, or an image using appropriate
            softwares.

        """
        self.plot_obs(obs, reward, done, timestamp)
        return pygame.surfarray.array3d(self.screen)

    def init_fig(self, fig, reward, done, timestamp):
        self.init_pygame()

        if not self.display_called:
            self.display_called = True
            self.screen.fill(self.background_color)
            pygame.display.set_caption('Grid2Op Renderer')  # Window title

        force, has_quit = self._event_looper(force=False)
        while force:
            force, has_quit = self._event_looper(force=force)
            pygame.time.wait(250)  # it's in ms

        if has_quit:
            self._quit_and_close()

        if reward is not None:
            self.cum_reward += reward
            self.nb_timestep += 1

        # The game is not paused anymore (or never has been), I can render the next surface

        if self.time_last is not None and self._deactivate_display is False:
            tmp = time.time()  # in second
            if tmp - self.time_last < self.timestep_duration_seconds:
                nb_sec_wait = int(1000 * (self.timestep_duration_seconds - (tmp - self.time_last)))
                pygame.time.wait(nb_sec_wait)  # it's in ms
            self.time_last = time.time()
        else:
            self.time_last = time.time()
        self.screen.fill(self.background_color)

        if done is not None:
            if not done:
                # draw the generic information on the right part
                self._draw_generic_info(reward, done, timestamp)
            else:
                # inform user that it's over
                self._draw_final_information(reward, done, timestamp)

    def _post_process_obs(self, fig, reward, done, timestamp, subs, lines, loads, gens, topos):
        """

         .. warning:: /!\\\\ This class is deprecated /!\\\\

        In canse of plotply, fig is whether the player press "quit" or not

        Parameters
        ----------
        fig
        subs
        lines
        loads
        gens
        topos

        Returns
        -------

        """
        self._draw_final_information(reward, done, timestamp)
        pygame.display.flip()
        if self._deactivate_display is False:
            if done:
                key_pressed = False
                while not key_pressed:
                    key_pressed, has_quit = self._press_key_to_quit()
                    # TODO that with fps !!!
                    pygame.time.wait(250)  # it's in ms
                self._quit_and_close()

    def _draw_generic_info(self, reward=None, done=None, timestamp=None):
        if reward is not None:
            text_label = "Instantaneous reward: {:.1f}".format(reward)
            text_graphic = self.font.render(text_label, True, self.color_text)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 100))
            text_label = "Cumulated reward: {:.1f}".format(self.cum_reward)
            text_graphic = self.font.render(text_label, True, self.color_text)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 130))
            text_label = "Number timesteps: {:.1f}".format(self.nb_timestep)
            text_graphic = self.font.render(text_label, True, self.color_text)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 160))

        if timestamp is not None:
            text_label = "Date : {:%Y-%m-%d %H:%M}".format(timestamp)
            text_graphic = self.font.render(text_label, True, self.color_text)
            self.screen.blit(text_graphic, (self.window_grid[0]+100, 200))

    def _draw_subs_one_sub(self, fig, sub_id, center, this_col, text):
        pygame.draw.circle(self.screen,
                           self.color_text,
                           [int(el) for el in center],
                           int(self.radius_sub),
                           2)

        text_graphic = self.font.render(text, True, this_col)
        self._aligned_text(center, text_graphic, center)

    def _get_default_cmap(self, normalized_val):
        # step 0: compute thickness and color
        max_val = 1.
        ratio_ok = 0.7
        start_red = 0.5
        amount_green = 235 - int(235. * (normalized_val / (max_val * ratio_ok))**4)
        amount_red = int(255 * (normalized_val/ (max_val * ratio_ok))**4)

        # if normalized_val < ratio_ok * max_val:
        #    amount_green = 100 - int(100. * normalized_val / (max_val * ratio_ok))
        # if normalized_val > ratio_ok:
        #    tmp = (1.0 * (normalized_val - ratio_ok) / (max_val - ratio_ok))**2
        #    amount_red += int(235. * tmp)
        # print("normalized_val {}, amount_red {}".format(normalized_val, amount_red))

        # fix to prevent pygame bug
        if amount_red < 0:
            amount_red = int(0)
        elif amount_red > 255:
            amount_red = int(255)
        if amount_green < 0:
            amount_green = int(0)
        elif amount_green > 255:
            amount_green = int(255)
        amount_red = int(amount_red)
        amount_green = int(amount_green)
        color = pygame.Color(amount_red, amount_green, 20)
        return color

    def _draw_powerlines_one_powerline(self, fig, l_id, pos_or, pos_ex, status, value, txt_, or_to_ex, this_col):
        text_graphic = self.font.render(txt_, True, this_col)
        pos_txt = [int((pos_or[0] + pos_ex[0]) * 0.5), int((pos_or[1] + pos_ex[1]) * 0.5)]
        how_center = "center|center"
        self._aligned_text(how_center, text_graphic, pos_txt)

        if not status:
            # line is disconnected
            _draw_dashed_line(self.screen, this_col, pos_or, pos_ex)
        else:
            # line is connected

            width = 1
            if value > self.rho_max:
                width = 4
            elif value > 1.:
                width = 3
            elif value > 0.9:
                width = 2
            width += 3

            # step 1: draw the powerline with right color and thickness
            pygame.draw.line(self.screen, this_col, pos_or, pos_ex, width)

            # step 2: draw arrows indicating current flows
            _draw_arrow(self.screen, this_col, pos_or, pos_ex,
                        or_to_ex,
                        num_arrows=width,
                        width=width)

    def _aligned_text(self, pos, text_graphic, pos_text):
        if isinstance(pos_text, complex):
            pos_x = pos_text.real
            pos_y = pos_text.imag
        else:
            pos_x, pos_y = pos_text

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
        elif pos == "center|center":
            pos_x -= width // 2
            pos_y -= height // 2
        self.screen.blit(text_graphic, (pos_x, pos_y))

    def _draw_loads_one_load(self, fig, l_id, pos_load, txt_, pos_end_line, pos_load_sub, how_center, this_col):
        width = 2
        pygame.draw.line(self.screen, this_col, pos_load_sub, (pos_end_line.real, pos_end_line.imag), width)
        text_graphic = self.font.render(txt_, True, this_col)
        self._aligned_text(how_center, text_graphic, pos_load)

    def _draw_gens_one_gen(self, fig, g_id, pos_gen, txt_, pos_end_line, pos_gen_sub, how_center, this_col):
        width = 2
        pygame.draw.line(self.screen, this_col, pos_gen_sub, (pos_end_line.real, pos_end_line.imag), width)
        text_graphic = self.font.render(txt_, True, this_col)
        self._aligned_text(how_center, text_graphic, pos_gen)
        return None

    def _draw_topos_one_sub(self, fig, sub_id, buses_z, elements, bus_vect):
        colors = [pygame.Color(255, 127, 14), pygame.Color(31, 119, 180)]

        # I plot the buses
        for bus_id, z_bus in enumerate(buses_z):
            pygame.draw.circle(self.screen,
                               colors[bus_id],
                               [int(z_bus.real), int(z_bus.imag)],
                               int(self.bus_radius),
                               0)

        # i connect every element to the proper bus with the proper color
        for el_nm, dict_el in elements.items():
            this_el_bus = bus_vect[dict_el["sub_pos"]] -1
            if this_el_bus >= 0:
                pygame.draw.line(self.screen,
                                 colors[this_el_bus],
                                 [int(dict_el["z"].real), int(dict_el["z"].imag)],
                                 [int(buses_z[this_el_bus].real), int(buses_z[this_el_bus].imag)],
                                 2)
        return []
