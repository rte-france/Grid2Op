# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy, re
import io
import numpy as np
import matplotlib as mpl
from dataclasses import dataclass, fields
from typing import Tuple, Literal, Union, List
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, is_color_like, BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Patch

from grid2op.Observation import ObservationSpace, BaseObservation
from grid2op.PlotGrid.BasePlot import BasePlot
from grid2op.PlotGrid.PlotUtil import PlotUtil as pltu
from grid2op.PlotGrid.config import *  # all colors
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class LoadPatch(patches.RegularPolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, n_vertices:int=3, **kwargs):
        patches.RegularPolygon.__init__(self, *args, numVertices=n_vertices, **kwargs)


class StoragePatch(patches.RegularPolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, n_vertices:int=4, **kwargs):
        patches.RegularPolygon.__init__(self, *args, numVertices=n_vertices, **kwargs)

class GenPatch(patches.RegularPolygon):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    Empty class to handle the legend
    """

    def __init__(self, *args, n_vertices:int=5, **kwargs):
        patches.RegularPolygon.__init__(self, *args, numVertices=n_vertices, **kwargs)

# TODO Refactor this class to make possible some calls like
#      plotmatplot.plot_info(...).plot_gentype(...) is possible

# TODO add some transparency when coloring=... is used in plot_info
# TODO code the load part in the plot_info

@dataclass
class PatchSettings:
    radius:int
    name:bool
    id:bool
    color_attr:Union[str,None]
    face_color:str
    cmap:Colormap
    cnorm:Union[Normalize,None]
    active_cnorm:Union[Normalize,None]
    vertices:Union[int,None] # No. of Points on Polygon
    txt_color:str
    line_color:Union[str,None]
    line_width:Union[float,None]
    display_value:bool
    display_name:bool

@dataclass
class LineSettings:
    busbar_radius:int
    name:str
    id:bool
    color_attr:Union[str,None]
    width:Union[float,None]
    cmap:Colormap
    cnorm:Union[Normalize,None]
    active_cnorm:Union[Normalize,None]
    busbar_cmap:Colormap
    arrow_len:int = 10
    arrow_width:float = 10.0

@dataclass
class Settings:
    sub:PatchSettings
    load:PatchSettings
    gen:PatchSettings
    storage:PatchSettings
    line:LineSettings

class PlotMatplot(BasePlot):  # pragma: no cover
    """
    This class uses the python library "matplotlib" to draw the powergrid.

    Arguments
    ----------
    observation_space: ``grid2op.Observation.ObservationSpace``
    width: ``int`` (default: 1280)
    height: ``int`` (default: 720)
    grid_layout ``dict`` (default: None) 
    dpi: ``int`` (default: 96)
    scale: ``float`` (default: 2000.0)
    
    # >> Substation <<
    sub_radius: ``int`` (default: 15)
    sub_color: ``str`` (default: "blue")
    
    # >> Load <<
    load_radius: ``int`` (default: 8)
    load_name: ``bool`` (default: False)
    load_id: ``bool`` (default: False)
    load_vertices: ``int`` (default: 3) # No. edges load polygon
    load_color_attr: ``str|None`` (default: None)
    load_color: ``Colormap|str|list`` (default: "orange")
    
    # >> Generator <<
    gen_radius: ``int`` (default: 8)
    gen_name: ``bool`` (default: False)
    gen_id: ``bool`` (default: False)
    gen_vertices: ``int`` (default: 5) # No. edges gen polygon
    gen_color_attr: ``Literal["gen_type", "prod_p"]`` (default: "prod_p")
    gen_color: ``Colormap|str|list`` (default: "green") 
    
    # >> Storage <<
    storage_radius: ``int`` (default: 8)
    storage_name: ``bool`` (default: False)
    storage_id: ``bool`` (default: False)
    storage_vertices: ``int`` (default: 4) # No. edges ESS polygon
    storage_color_attr: ``str|None`` (default: None)
    storage_color: ``Colormap|str|list`` (default: "purple")
    
    # >> Line <<
    line_name: ``bool`` (default: False)
    line_id: ``bool`` (default: False)
    line_color_norm: ``Normalize`` (default: None) # BoundaryNorm([0.0, 0.5, 1.0], ncolors=3, extend="max"),
    line_color_attr: ``Literal["rho", "a", "p", "v", "a_or", "p_or", "v_or"]`` (default: "rho")
    line_color: ``Colormap|str|list`` (default: ["blue", "orange", "red"]) # Cmap name, or list of colours
    
    # >> Busbar <<
    bus_radius: ``int`` (default: 6)
    bus_color: ``Colormap|str|list`` (default: ["black", "red", "lime"])


    Attributes
    ----------

    width: ``int``
        Width of the figure in pixels
    height: ``int``
        Height of the figure in pixel
    dpi: ``int``
        Dots per inch, to convert pixels dimensions into inches
    settings: ``Settings``
        Settings for the size, coloring and information displayed for loads, generators, storage, and powerlines.
    
    Examples
    --------
    You can use it this way:

    .. code-block:: python

        import grid2op
        from grid2op.PlotGrid import PlotMatplot
        env = grid2op.make("l2rpn_case14_sandbox")
        plot_helper = PlotMatplot(env.observation_space)

        # and now plot an observation (for example)
        obs = env.reset()
        fig = plot_helper.plot_obs(obs)
        fig.show()

        # more information about it on the `getting_started/8_PlottingCapabilities.ipynb` notebook of grid2op

    """
    
    line_attr_map = {"a":"a_or", "p":"p_or", "v":"v_or"}
    load_attr_map = {"p":"load_p", "v":"load_v", "q":"load_q"}
    gen_attr_map = {"p":"prod_p", "v":"prod_v", "q":"prod_q", "type":"gen_type"}
    storage_attr_map = {"p":"storage_power", "c":"storage_charge", "theta":"storage_theta"}

    def __init__(self, observation_space:ObservationSpace,
                 width:int=1280, height:int=720,
                 grid_layout:dict=None, 
                 dpi:int=96, scale:float=2000.0,
                 
                 # >> Substation <<
                 sub_radius:int=15, sub_color:str = "blue",
                 
                 # >> Load <<
                 load_radius:int=8, load_name:bool=False,
                 load_id:bool=False, load_vertices:int=3, # No. edges load polygon
                 load_color_norm:Union[Normalize,List[float]] = None,
                 load_color_attr:Literal["p","v"] = "p",
                 load_color:Union[Colormap,str,list] = "orange",
                 
                 # >> Generator <<
                 gen_radius:int=8, gen_name:bool=False,
                 gen_id:bool=False, gen_vertices:int=5, # No. edges gen polygon
                 gen_color_norm:Union[Normalize,List[float]] = None,
                 gen_color_attr:Literal["type", "p", "q", "v"]= "p",
                 gen_color:Union[Colormap,str,list] = "green", 
                 
                 # >> Storage <<
                 storage_radius:int=8, storage_name:bool=False,
                 storage_id:bool=False, storage_vertices:int=4, # No. edges ESS polygon
                 storage_color_norm:Union[Normalize,List[float]]  = None,
                 storage_color_attr:Literal["p", "theta", "charge"] = "p",
                 storage_color:Union[Colormap,str,list] = "purple",
                 
                 # >> Line <<
                 line_name:bool=False, line_id:bool=False,
                 line_width:float=1.0,
                 line_color_norm:Union[Normalize,List[float]]  = None,
                 line_color_attr:Literal["rho", "p", "a", "v"]="rho",
                 line_color:Union[Colormap,str,list]=["blue", "orange", "red"], # Cmap name, or list of colours
                 
                 # >> Busbar <<
                 bus_radius:int=6,
                 bus_color:Union[Colormap,str,list]=["black", "red", "lime"],
        ):
        self.dpi = dpi
        super().__init__(observation_space, width, height, scale, grid_layout)
        
        _sub_settings = PatchSettings(sub_radius, False, False, None,
                                     "w", self._convert_colors_to_cmap(sub_color), 
                                     None, None,
                                     None, "black", "black", 1, 
                                     False, True)
        
        load_color_attr = PlotMatplot.load_attr_map.get(load_color_attr, load_color_attr)
        _load_settings = PatchSettings(load_radius, load_name, load_id, load_color_attr,
                                       "w", self._convert_colors_to_cmap(load_color), 
                                       self._convert_values_to_norm(load_color_norm), None,
                                       load_vertices,
                                       "black", "black", 1, 
                                       False, True)
        self._load_patch = self._load_patch_default
        
        gen_color_attr = PlotMatplot.gen_attr_map.get(gen_color_attr, gen_color_attr)
        _gen_settings = PatchSettings(gen_radius, gen_name, gen_id, gen_color_attr,
                                      "w", self._convert_colors_to_cmap(gen_color), 
                                      self._convert_values_to_norm(gen_color_norm), None,
                                      gen_vertices,
                                      "black", "black", 1,
                                      True, True)
        self._gen_patch = self._gen_patch_default
        
        storage_color_attr = PlotMatplot.storage_attr_map.get(storage_color_attr, storage_color_attr)
        _storage_settings = PatchSettings(storage_radius, storage_name, storage_id, storage_color_attr,
                                          "w", self._convert_colors_to_cmap(storage_color), 
                                          self._convert_values_to_norm(storage_color_norm), None,
                                          storage_vertices,
                                          "black", "black", 1,
                                          False, True)
        self._storage_patch = self._storage_patch_default
        
        # Line Color Normalizaion (if None, will normalize by attr when obs is passed)
        line_color_attr = PlotMatplot.line_attr_map.get(line_color_attr, line_color_attr)
        if line_color_attr == "rho" and line_color_norm is None: # 1.11.0: Backwards compatability
            line_color_norm = BoundaryNorm([0.0, 0.5, 1.0], ncolors=3, extend="max")
        
        _line_settings = LineSettings(bus_radius, line_name, line_id,
                                      line_color_attr, line_width, 
                                      self._convert_colors_to_cmap(line_color),
                                      self._convert_values_to_norm(line_color_norm), None,
                                      self._convert_colors_to_cmap(bus_color))

        self.settings = Settings(_sub_settings, _load_settings, _gen_settings, 
                                  _storage_settings, _line_settings)
        self.orig_settings = copy.deepcopy(self.settings)

        self.xlim = [0, 0]
        self.xpad = 5
        self.ylim = [0, 0]
        self.ypad = 5

        # For easize manipulation
        self.legend = None
        self.figure:Union[Figure,None] = None
        self.blitting:bool = False
        
    def _convert_colors_to_cmap(self, color_scheme:Union[str,list,Colormap], nb_color:int=-1) -> Colormap:
        """
        Converts a colormap name (as str) or a list of color-like objects to a 
        'matpolitlib.colors.Colormap'. 
        
        Optionally will refit the colormap into a Discretized linear segemented color-map
        if nb-color is provided.

        Args:
            color_scheme (str | list | Colormap): Specific color, name of colormap, or list of color-like objects.
            nb_color (int, optional): Number of discrete color categories. Defaults to -1 (no affect).

        Raises:
            ValueError: Colour scheme is not valid
        
        Returns:
            Colormap: A Matplotlib Colormap which when called converts a float to a color 4-tuple (r,g,b,a). 
        """
        cmap = None
        if isinstance(color_scheme, Colormap):
            cmap =  color_scheme
        elif isinstance(color_scheme, str):
            if color_scheme in mpl.colormaps: # String is a colormap
                cmap =  mpl.colormaps[color_scheme]
            elif is_color_like(color_scheme): # String is a color
                cmap = LinearSegmentedColormap.from_list("", [color_scheme]*2, N=2)
            else:
                raise ValueError(f"String: '{color_scheme}' is not a valid color nor colormap name.")
        elif isinstance(color_scheme, list):
            if len(color_scheme) == 1: # Single Value
                color_scheme.append(color_scheme[0])
            cmap = LinearSegmentedColormap.from_list("", color_scheme, N=len(color_scheme))
        else: 
            raise ValueError(f"Colour Scheme not found: '{color_scheme}'")
        
        if nb_color > 0:
            return LinearSegmentedColormap.from_list("",
                [cmap(i / nb_color) for i in range(nb_color)], N=nb_color)
        else:
            return cmap
        
    def _convert_values_to_norm(self, color_norm:Union[List[float],Normalize,None]) -> Normalize:
        """
        Converts a list of floats to a color Normalizer in Matplotlib. If two valeus are provided
        the normalization is done linearly between them. If more are provided, a Discete boundary
        norm is used instead. If the normalization is defined manually, it is used as is.

        Args:
            color_norm (List[float]|Normalize|None): Specific color normalization.

        Returns:
            Normalize: A Matplotlib Normalizer which when called converts a float to its normalized value.
        """
        if isinstance(color_norm, list):
            if len(color_norm) == 2: # Continuous between [min, max]
                color_norm = Normalize(vmin=np.min(np.abs(color_norm)),
                                                         vmax= np.max(np.abs(color_norm)))
            else: # Linear Discrete
                color_norm= BoundaryNorm(np.sort(np.abs(color_norm)), ncolors=len(color_norm), extend="max")
        return color_norm
        
    def _set_active_color_norm(self, obs:Union[BaseObservation,ObservationSpace], norm:Union[Normalize,None],
                               attr:Union[str,None], settings:Union[PatchSettings,LineSettings]):
        """
        Set the currently active color normalization, which overwrites (in-place) the settings for a specific patch.

        Args:
            obs (BaseObservation|ObservationSpace): Current observation (to be visualized) or Observation Space (static)
            norm (Normalize | None): Color normalization process in Matplotlib
            attr (str | None): Name of attribute to color by the patches by
            settings (PatchSettings | Line Settings): Settings to modify, belonging to a specific patch category.
        """
        attr = attr if attr is not None else settings.color_attr
        if isinstance(obs, ObservationSpace):
            norm = None
        elif norm is None and settings.cnorm is None and attr is not None:
            try:
                values = getattr(obs, attr)
            except Exception:
                values = getattr(obs, f"{attr}_or")
            values = np.abs(values) # Direction does not matter
            if values.size > 0:
                norm = Normalize(vmin=np.min(values), vmax=np.max(values), clip=True)
        elif norm is None:
            norm = copy.deepcopy(settings.cnorm)
        settings.active_cnorm = norm

    def _set_active_color_norms(self, obs:Union[BaseObservation,ObservationSpace], **kwargs):
        """
        Set the active color normalization for all patches (load, gen, storage, and lines). Behaviour depends
        on whether kwargs overwrites any of the normalizations for this specific observation.
        """
        for field in fields(Settings):
            self._set_active_color_norm(obs, kwargs.get(f"{field.name}_color_norm", None),
                                        kwargs.get(f"{field.name}_color_attr", None), getattr(self.settings, field.name))
    
    def _gen_patch_default(self, xy:Tuple[float, float], radius:float,
                           edgecolor:Union[object,None], facecolor:Union[object,None]):
        """Default patch used to draw generators"""
        return GenPatch(
            xy, radius=radius,
            edgecolor=edgecolor, facecolor=facecolor,
            n_vertices=self.settings.gen.vertices,
            linewidth=self.settings.gen.line_width,
        )

    def _load_patch_default(self, xy:Tuple[float, float], radius:float,
                            edgecolor:Union[object,None], 
                            facecolor:Union[object,None]) -> Patch:
        """Default patch used to draw loads"""
        return LoadPatch(
            xy, radius=radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=self.settings.load.line_width,
            n_vertices=self.settings.load.vertices,
        )
    
    def _storage_patch_default(self, xy:Tuple[float, float], radius:float,
                               edgecolor:Union[object,None],
                               facecolor:Union[object,None]) -> Patch:
        """Default patch used to draw Storage systems"""
        return StoragePatch(
            xy, radius=radius,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linewidth=self.settings.storage.line_width,
            n_vertices=self.settings.storage.vertices,
        )

    def _v_textpos_from_dir(self, diry:float) -> str:
        if diry > 0:
            return "bottom"
        else:
            return "top"

    def _h_textpos_from_dir(self, dirx:float) -> str:
        if dirx == 0:
            return "center"
        elif dirx > 0:
            return "left"
        else:
            return "right"

    def create_figure(self) -> Figure:
        """
        Lazy loading of Graphics library (reduces loading time)
        Also needed because Matplotlib has a weird impact on argparse.
        """
        import matplotlib.pyplot as plt
        w_inch = self.width / self.dpi
        h_inch = self.height / self.dpi
        f = plt.figure(figsize=(w_inch, h_inch), dpi=self.dpi)
        self.ax = f.subplots()
        f.canvas.draw()
        self.blitting = f.canvas.supports_blit
        return f

    def clear_figure(self, figure:Figure):
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        figure.clear()
        self.ax = figure.subplots()

    def convert_figure_to_numpy_HWC(self, figure:Figure) -> np.ndarray:
        w, h = figure.get_size_inches() * figure.dpi
        w, h = int(w), int(h)
        buf = io.BytesIO()
        figure.canvas.print_raw(buf)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img_arr = np.reshape(img_arr, (h, w, 4))
        return img_arr

    def _draw_substation_txt(self, pos_x:float, pos_y:float, text:str):
        self.ax.text(
            pos_x, pos_y, text,
            color=self.settings.sub.txt_color,
            horizontalalignment="center",
            verticalalignment="center"
        )

    def _draw_substation_circle(self, pos_x:float, pos_y:float, 
                                sub_edgecolor:Union[object,None]):
        patch = patches.Circle(
            (pos_x, pos_y),
            radius=self.settings.sub.radius,
            facecolor=self.settings.sub.face_color,
            edgecolor=sub_edgecolor,
        )
        self.ax.add_patch(patch)

    def draw_substation(self, figure:Figure, observation:BaseObservation,
                        sub_id:int, sub_name:str, 
                        pos_x:int, pos_y:int):
        self.xlim[0] = min(self.xlim[0], pos_x - self.settings.sub.radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self.settings.sub.radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self.settings.sub.radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self.settings.sub.radius)
        
        edge_color = self.settings.sub.cmap(0.5)
        self._draw_substation_circle(pos_x, pos_y, edge_color)
        if self.settings.sub.display_name:
            self._draw_substation_txt(pos_x, pos_y, str(sub_id))

    def _draw_load_txt(self, pos_x:float, pos_y:float,
                       sub_x:float, sub_y:float, text:str):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self.settings.gen.radius
        txt_y = pos_y + off_y * self.settings.gen.radius
        ha = self._h_textpos_from_dir(dir_x)
        va = self._v_textpos_from_dir(dir_y)
        self.ax.text(
            txt_x, txt_y, text,
            color=self.settings.load.txt_color,
            ha=ha, va=va,
            fontsize="small",
        )

    def _draw_load_name(self, pos_x:float, pos_y:float, txt:str):
        self.ax.text(
            pos_x, pos_y, txt,
            color=self.settings.load.txt_color,
            va="center", ha="center",
            fontsize="x-small",
        )

    def _draw_load_circle(self, pos_x:float, pos_y:float,
                          load_edgecolor:Union[object,None]):
        patch = self._load_patch(
            (pos_x, pos_y),
            radius=self.settings.load.radius,
            facecolor=self.settings.load.face_color,
            edgecolor=load_edgecolor,
        )
        self.ax.add_patch(patch)

    def _draw_load_line(self, pos_x:float, pos_y:float, 
                        sub_x:float, sub_y:float):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, 
            color=self.settings.load.line_color,
            lw=self.settings.load.line_width
        )
        self.ax.add_patch(patch)

    def draw_load(self, figure:Figure, observation:BaseObservation,
                  load_id:int, load_name:str, load_bus:int,
                  load_value:float, load_unit:str,
                  pos_x:int, pos_y:int, sub_x:int, sub_y:int):
        self.xlim[0] = min(self.xlim[0], pos_x - self.settings.load.radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self.settings.load.radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self.settings.load.radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self.settings.load.radius)
        self._draw_load_line(pos_x, pos_y, sub_x, sub_y)

        color_attr = "" if self.settings.load.color_attr is None else self.settings.load.color_attr
        if self.settings.load.active_cnorm is None:
            color_value = 0.5
        elif hasattr(observation, color_attr):
            color_value = self.settings.load.active_cnorm(np.abs(getattr(observation, self.settings.load.color_attr)[load_id]))
        else:
            color_value = (self.settings.load.active_cnorm.vmax - self.settings.load.active_cnorm.vmin) / 2
        edge_color = self.settings.load.cmap(color_value)
        self._draw_load_circle(pos_x, pos_y, load_edgecolor=edge_color)
        load_txt = ""
        if self.settings.load.name:
            load_txt += f'"{load_name}":\n'
        if self.settings.load.id:
            load_txt += f"id: {load_id}\n"
        if load_value is not None:
            load_txt += pltu.format_value_unit(load_value, load_unit)
        if load_txt:
            self._draw_load_txt(pos_x, pos_y, sub_x, sub_y, load_txt)
        if self.settings.load.display_name:
            self._draw_load_name(pos_x, pos_y, str(load_id))
        load_dir_x, load_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        self._draw_busbar(sub_x, sub_y, load_dir_x, load_dir_y, load_bus)

    def _draw_gen_txt(self, pos_x:float, pos_y:float,
                      sub_x:float, sub_y:float, text:str):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self.settings.gen.radius
        txt_y = pos_y + off_y * self.settings.gen.radius
        ha = self._h_textpos_from_dir(dir_x)
        va = self._v_textpos_from_dir(dir_y)
        self.ax.text(
            txt_x, txt_y, text,
            color=self.settings.gen.txt_color,
            ha=ha, va=va, wrap=True,
            fontsize="small",
        )

    def _draw_gen_circle(self, pos_x:float, pos_y:float,
                         gen_edgecolor:Union[object,None]):
        patch = self._gen_patch(
            (pos_x, pos_y),
            radius=self.settings.gen.radius,
            edgecolor=gen_edgecolor,
            facecolor=self.settings.gen.face_color,
        )
        self.ax.add_patch(patch)

    def _draw_gen_line(self, pos_x:float, pos_y:float,
                       sub_x:float, sub_y:float):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, 
            color=self.settings.gen.line_color, 
            lw=self.settings.load.line_width
        )
        self.ax.add_patch(patch)

    def _draw_gen_name(self, pos_x:float, pos_y:float, txt:str):
        self.ax.text(
            pos_x, pos_y, txt,
            color=self.settings.gen.txt_color,
            va="center", ha="center",
            fontsize="x-small",
        )

    def draw_gen(self, figure:Figure, observation:BaseObservation,
                 gen_id:int, gen_name:str, gen_bus:int,
                 gen_value:float, gen_unit:str,
                 pos_x:int, pos_y:int, sub_x:int, sub_y:int):
        self.xlim[0] = min(self.xlim[0], pos_x - self.settings.gen.radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self.settings.gen.radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self.settings.gen.radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self.settings.gen.radius)

        color_attr = "" if self.settings.gen.color_attr is None else self.settings.gen.color_attr

        if self.settings.gen.color_attr == "prod_p":
            vmin = observation.gen_pmin[gen_id] if observation.gen_pmin is not None else None
            vmax = observation.gen_pmax[gen_id] if observation.gen_pmax is not None else None
            norm = Normalize(vmin=vmin, vmax=vmax)
            try:
                color_value = norm(observation.prod_p[gen_id])
            except Exception: # Only static info available, don't show generator output
                color_value = norm(np.nan)
        elif self.settings.gen.color_attr == "gen_type":
            norm = Normalize(vmin=0.0, vmax=len(TYPE_GEN))
            color_value = norm(TYPE_GEN[observation.gen_type[gen_id]])
        elif self.settings.gen.active_cnorm is None:
            color_value = Normalize(vmin=0.0, vmax=1.0)(0.5)
        elif hasattr(observation, color_attr):
            color_value = self.settings.gen.active_cnorm(np.abs(getattr(observation, self.settings.gen.color_attr)[gen_id]))
        else:
            color_value = (self.settings.gen.active_cnorm.vmax - self.settings.gen.active_cnorm.vmin) / 2
        
        gen_edge_color = self.settings.gen.cmap(color_value) if np.isfinite(color_value) else "green"
        self._draw_gen_line(pos_x, pos_y, sub_x, sub_y)
        self._draw_gen_circle(pos_x, pos_y, gen_edge_color)
        gen_txt = ""
        if self.settings.gen.name:
            gen_txt += f'"{gen_name}":\n'
        if self.settings.gen.id:
            gen_txt += f"id: {gen_id}\n"
        if np.isfinite(color_value):
            if gen_value is not None and self.settings.gen.display_value:
                gen_txt += pltu.format_value_unit(gen_value, gen_unit)
        if gen_txt:
            self._draw_gen_txt(pos_x, pos_y, sub_x, sub_y, gen_txt)
        if self.settings.gen.display_name:
            self._draw_gen_name(pos_x, pos_y, str(gen_id))
        gen_dir_x, gen_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        self._draw_busbar(sub_x, sub_y, gen_dir_x, gen_dir_y, gen_bus)
    
    def _draw_storage_txt(self, pos_x:float, pos_y:float,
                          sub_x:float, sub_y:float, text:str):
        self._draw_load_txt(pos_x, pos_y, sub_x, sub_y, text)
        
    def _draw_storage_name(self, pos_x:float, pos_y:float, txt:str):
        self._draw_load_name(pos_x, pos_y, txt)

    def draw_storage(self, figure:Figure, observation:BaseObservation,
                     storage_id:int, storage_name:str, storage_bus:int,
                     storage_value:float, storage_unit:str,
                     pos_x:int, pos_y:int, sub_x:int, sub_y:int):
        self.xlim[0] = min(self.xlim[0], pos_x - self.settings.load.radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self.settings.load.radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self.settings.load.radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self.settings.load.radius)
        self._draw_storage_line(
            pos_x, pos_y, sub_x, sub_y
        )  # line from the storage to the substation

        color_attr = "" if self.settings.storage.color_attr is None else self.settings.storage.color_attr
        if self.settings.storage.active_cnorm is None:
            color_value = 0.5
        elif hasattr(observation, color_attr):
            color_value = self.settings.storage.active_cnorm(np.abs(getattr(observation, self.settings.storage.color_attr)[storage_id]))
        else:
            color_value = (self.settings.storage.active_cnorm.vmax - self.settings.storage.active_cnorm.vmin) / 2
        edge_color = self.settings.storage.cmap(color_value)
        self._draw_storage_circle(pos_x, pos_y, stor_edgecolor=edge_color)  # storage element

        storage_txt = ""
        if self.settings.storage.name:
            storage_txt += '"{}":\n'.format(storage_name)
        if self.settings.storage.id:
            storage_txt += "id: {}\n".format(storage_id)
        if storage_value is not None:
            storage_txt += pltu.format_value_unit(storage_value, storage_unit)
        if storage_txt:
            self._draw_storage_txt(pos_x, pos_y, sub_x, sub_y, storage_txt)
        if self.settings.storage.display_name:
            self._draw_storage_name(pos_x, pos_y, str(storage_id))
        storage_dir_x, storage_dir_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        self._draw_busbar(sub_x, sub_y, storage_dir_x, storage_dir_y, storage_bus)

    def _draw_storage_circle(self, pos_x:float, pos_y:float,
                             stor_edgecolor:Union[object,None]):
        patch = self._storage_patch(
            (pos_x, pos_y),
            radius=self.settings.storage.radius,
            facecolor=self.settings.storage.face_color,
            edgecolor=stor_edgecolor,
        )
        self.ax.add_patch(patch)

    def _draw_storage_line(self, pos_x:float, pos_y:float, sub_x:float, sub_y:float):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_x, pos_y), (sub_x, sub_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=self.settings.storage.line_color, 
            lw=self.settings.storage.line_width
        )
        self.ax.add_patch(patch)

    def _draw_busbar(self, pos_x:float, pos_y:float, 
                          norm_dir_x:float, norm_dir_y:float, bus_id:int):
        center_x = pos_x + norm_dir_x * self.settings.sub.radius
        center_y = pos_y + norm_dir_y * self.settings.sub.radius
        norm = Normalize(vmin=0, vmax=self.observation_space.n_busbar_per_sub)
        face_color = self.settings.line.busbar_cmap(norm(bus_id))
        patch = patches.Circle(
            (center_x, center_y), facecolor=face_color,
            radius=self.settings.line.busbar_radius 
        )
        self.ax.add_patch(patch)

    def _draw_powerline_txt(self, pos_or_x:float, pos_or_y:float,
                            pos_ex_x:float, pos_ex_y:float, text:str):
        pos_x, pos_y = pltu.middle_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        off_x, off_y = pltu.orth_norm_from_points(
            pos_or_x, pos_or_y, pos_ex_x, pos_ex_y
        )
        txt_x = pos_x + off_x * (self.settings.load.radius / 2)
        txt_y = pos_y + off_y * (self.settings.load.radius / 2)
        ha = self._h_textpos_from_dir(off_x)
        va = self._v_textpos_from_dir(off_y)
        self.ax.text(
            txt_x, txt_y, text,
            color=self.settings.gen.txt_color,
            ha=ha, va=va,
            fontsize="small",
        )

    def _draw_powerline_line(self, pos_or_x:float, pos_or_y:float,
                             pos_ex_x:float, pos_ex_y:float,
                             color:Union[object,None], line_style:str):
        codes = [Path.MOVETO, Path.LINETO]
        verts = [(pos_or_x, pos_or_y), (pos_ex_x, pos_ex_y)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, color=color, ls=line_style,
            lw=self.settings.line.width,
        )
        self.ax.add_patch(patch)

    def _draw_powerline_arrow(self, pos_or_x:float, pos_or_y:float,
                              pos_ex_x:float, pos_ex_y:float,
                              color:Union[object,None], watt_value:float):
        sign = 1.0 if watt_value > 0.0 else -1.0
        off = 1.0 if watt_value > 0.0 else 2.0
        dx, dy = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        lx = dx * self.settings.line.arrow_len
        ly = dy * self.settings.line.arrow_len
        arr_x = pos_or_x + dx * self.settings.sub.radius+ off * lx
        arr_y = pos_or_y + dy * self.settings.sub.radius+ off * ly
        patch = patches.FancyArrow(
            arr_x, arr_y, # (x,y)
            sign * lx, sign * ly, # (dx, dy)
            length_includes_head=True,
            head_length=self.settings.line.arrow_len,
            head_width=self.settings.line.arrow_width,
            edgecolor=color, facecolor=color,
        )
        self.ax.add_patch(patch)

    def assign_line_palette(self, color_scheme:Union[Colormap,str,list]="YlOrRd", nb_color:int=-1):
        """
        Assign a new color palette when you want to plot information on the powerline.

        Parameters
        ----------
        color_scheme: ``Colormap|str|list``
            Name of the Maplotlib.plyplot palette to use (name forwarded to `plt.get_cmap(palette_name)`)
        
        nb_color: ``int``
            Number of color to use.

        Examples
        -------
        .. code-block:: python

            # color a grid based on the value of the thermal limit
            plot_helper.assign_line_palette(nb_color=100)

            # plot this grid
            _ = plot_helper.plot_info(line_values=env.get_thermal_limit(), line_unit="A", coloring="line")

            # restore the default coloring (["blue", "orange", "red"])
            plot_helper.restore_line_palette()

        Notes
        -----
        Some palette are available there `colormaps <https://matplotlib.org/tutorials/colors/colormaps.html>`_

        """
        self.settings.line.cmap = self._convert_colors_to_cmap(color_scheme, nb_color=nb_color)

    def restore_line_settings(self):
        self.settings.line = copy.deepcopy(self.orig_settings.line)

    def assign_gen_palette(self, color_scheme:Union[Colormap,str,list]="YlOrRd", nb_color:int=-1,
                           increase_gen_size:Union[float,None]=None, gen_line_width:Union[float,None]=None):
        """
        Assign a new color palette when you want to plot information on the generator.

        Parameters
        ----------
        palette_name: ``str``
            Name of the Maplotlib.plyplot palette to use (name forwarded to `plt.get_cmap(palette_name)`)
        
        nb_color: ``int``
            Number of color to use

        increase_gen_size: ``float``
            Whether or not to increase the generator sizes (``None`` to disable this feature, 1 has no effect)

        gen_line_width: ``float``
            Increase the width of the generator (if not ``None``)

        Examples
        -------
        .. code-block:: python

            # color a grid based on the value of the thermal limit
            plot_helper.assign_gen_palette(nb_color=100)

            # plot this grid
            _ = plot_helper.plot_info(gen_values=env.gen_pmax, coloring="gen")

            # restore the default coloring (all green)
            plot_helper.restore_gen_palette()

        Notes
        -----
        Some palette are available there `colormaps <https://matplotlib.org/tutorials/colors/colormaps.html>`_

        """
        self.settings.gen.cmap = self._convert_colors_to_cmap(color_scheme, nb_color=nb_color)
        
        if increase_gen_size is not None: # The user changed the generator sizes
            self.settings.gen.radius = float(increase_gen_size) * self.orig_settings.gen.radius
        if gen_line_width is not None: # The user changed the generator line width
            self.settings.gen.line_width = float(gen_line_width)

    def restore_gen_settings(self):
        """Restore every properties of the default generator layout"""
        self.settings.gen = copy.deepcopy(self.orig_settings.gen)

    def restore_settings(self):
        """Restore every default setting"""
        self.settings = copy.deepcopy(self.orig_settings)
    
    def draw_powerline(self, figure:Figure, observation:BaseObservation,
                       line_id:int, line_name:str, connected:bool,
                       line_value:float, line_unit:str, 
                       or_bus:int, pos_or_x:int, pos_or_y:int,
                       ex_bus:int, pos_ex_x:int, pos_ex_y:int):
        color_attr = "" if self.settings.line.color_attr is None else self.settings.line.color_attr
        if self.settings.line.active_cnorm is None:
            color_value = 0.0
        elif hasattr(observation, self.settings.line.color_attr):
            color_value = self.settings.line.active_cnorm(np.abs(getattr(observation, color_attr)[line_id]))
        else:
            color_value = (self.settings.line.active_cnorm.vmax - self.settings.line.active_cnorm.vmin) / 2
        if np.isfinite(color_value):
            color = self.settings.line.cmap(color_value)
            line_style = "-" if connected else "--"
            self._draw_powerline_line(
                pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, line_style
            )
            # Deal with line text configurations
            txt = ""
            if self.settings.line.name:
                txt += '"{}"\n'.format(line_name)
            if self.settings.line.id:
                txt += "id: {}\n".format(str(line_id))
            if line_value is not None:
                txt += pltu.format_value_unit(line_value, line_unit)
            if txt:
                self._draw_powerline_txt(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, txt)

            or_dir_x, or_dir_y = pltu.norm_from_points(
                pos_or_x, pos_or_y, pos_ex_x, pos_ex_y
            )
            self._draw_busbar(pos_or_x, pos_or_y, or_dir_x, or_dir_y, or_bus)
            ex_dir_x, ex_dir_y = pltu.norm_from_points(
                pos_ex_x, pos_ex_y, pos_or_x, pos_or_y
            )
            self._draw_busbar(pos_ex_x, pos_ex_y, ex_dir_x, ex_dir_y, ex_bus)
            watt_value = observation.p_or[line_id]
            if color_value > 0.0 and np.abs(watt_value) >= 1e-7:
                self._draw_powerline_arrow(
                    pos_or_x, pos_or_y, pos_ex_x, pos_ex_y, color, watt_value
                )

    def _get_gen_legend(self):
        """super complex function to display the proper shape in the legend"""
        gen_legend_col = "green" # self.settings.gen.cmap(0.5)
        settings = self.settings.gen

        class GenObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = GenPatch(
                    xy=center,
                    radius=min(width, height),
                    facecolor=settings.face_color,
                    edgecolor=gen_legend_col,
                    transform=handlebox.get_transform(),
                    n_vertices=settings.vertices,
                )
                handlebox.add_artist(pp_)
                return pp_

        gen_legend = self._gen_patch(
            (0, 0),
            facecolor=self.settings.gen.face_color,
            edgecolor=gen_legend_col,
            radius=self.settings.gen.radius,
        )
        return gen_legend, GenObjectHandler()

    def _get_load_legend(self):
        """super complex function to display the proper shape in the legend"""
        load_legend_col = "orange" # self.settings.load.cmap(0.5)
        settings = self.settings.load

        class LoadObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = LoadPatch(
                    xy=center,
                    radius=min(width, height),
                    facecolor=settings.face_color,
                    edgecolor=load_legend_col,
                    transform=handlebox.get_transform(),
                    n_vertices=settings.vertices,
                )
                handlebox.add_artist(pp_)
                return pp_

        load_legend = self._load_patch(
            (0, 0),
            facecolor=self.settings.load.face_color,
            edgecolor=load_legend_col,
            radius=self.settings.load.radius,
        )
        return load_legend, LoadObjectHandler()

    def _get_storage_legend(self):
        """super complex function to display the proper shape in the legend"""
        storage_legend_col = "purple" # self.settings.storage.cmap(0.5)
        settings = self.settings.storage

        class StorageObjectHandler:
            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                xdescent, ydescent = handlebox.xdescent, handlebox.ydescent
                width, height = handlebox.width, handlebox.height
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                pp_ = StoragePatch(
                    xy=center,
                    radius=min(width, height),
                    facecolor=settings.face_color,
                    edgecolor=storage_legend_col,
                    transform=handlebox.get_transform(),
                    n_vertices=settings.vertices,
                )
                handlebox.add_artist(pp_)
                return pp_

        storage_legend = self._storage_patch(
            (0, 0),
            facecolor=self.settings.storage.face_color,
            edgecolor=storage_legend_col,
            radius=self.settings.storage.radius,
        )
        return storage_legend, StorageObjectHandler()

    def draw_legend(self, figure:Figure, observation:BaseObservation):
        title_str = observation.env_name
        if hasattr(observation, "month"):
            title_str = (f"{observation.day:02d}/{observation.month:02d} " +
                         f"{observation.hour_of_day:02d}:{observation.minute_of_hour:02d}")

        # generate the right legend for generator
        gen_legend, gen_handler = self._get_gen_legend()
        # generate the correct legend for load
        load_legend, load_handler = self._get_load_legend()
        # generate the correct legend for storage
        storage_legend, storage_handler = self._get_storage_legend()

        bus_norm = Normalize(vmin=-1, vmax=self.observation_space.n_busbar_per_sub)
        legend_help = [
            Line2D([0], [0], color="black", lw=1),
            Line2D([0], [0], color=self.settings.sub.cmap(0.5), lw=3),
            load_legend,
            gen_legend,
            storage_legend
        ] + [Line2D([0], [0], marker="o", color=self.settings.line.busbar_cmap(bus_norm(i))) 
             for i in range(-1, self.observation_space.n_busbar_per_sub)]
        self.legend = self.ax.legend(
            handles=legend_help,
            labels=["powerline", "substation", "load",
                    "generator", "storage", "no bus"] + 
                   [f"bus {i+1}" for i in range(self.observation_space.n_busbar_per_sub)],
            title=title_str,
            handler_map={
                GenPatch: gen_handler,
                LoadPatch: load_handler,
                StoragePatch: storage_handler,
            },
        )
        # Hide axis
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # Hide frame
        self.ax.set(frame_on=False)

        # save the figure
        self.figure = figure
    
    def plot_obs(self, observation:BaseObservation, *args, **kwargs):
        self._set_active_color_norms(observation, **kwargs)
        fig = super(PlotMatplot, self).plot_obs(observation, *args, **kwargs)
        if self.settings.gen.color_attr == "gen_type":
            self.add_legend_gentype()
        self._set_active_color_norms(observation) # Return to default
        return fig
    
    def plot_info(self, *args, **kwargs):
        fig = super(PlotMatplot, self).plot_info(*args, **kwargs)
        if self.settings.gen.color_attr == "gen_type":
            self.add_legend_gentype()
        self.restore_settings()
        return fig
    
    def plot_layout(self):
        """
        This function plot the layout of the grid, as well as the object. You will see the name of each elements and
        their id.
        """
        return self.plot_info(
            observation=None, figure=None, redraw=True, gen_values=np.zeros(self.observation_space.n_gen),
        )

    def plot_postprocess(self, figure:Figure, observation:BaseObservation, update:bool):
        if not update:
            xmin = self.xlim[0] - self.xpad
            xmax = self.xlim[1] + self.xpad
            self.ax.set_xlim(xmin, xmax)
            ymin = self.ylim[0] - self.ypad
            ymax = self.ylim[1] + self.ypad
            self.ax.set_ylim(ymin, ymax)
            figure.tight_layout()

    def _save_plot_charact(self):
        return copy.deepcopy(self.settings)

    def _restore_plot_charact(self, data):
        self.settings = data

    def plot_gen_type(self, increase_gen_size:float=1.5, gen_line_width:float=3):
        # save the sate of the generators config
        data = self._save_plot_charact()
        self.settings.gen.radius = self.settings.gen.radius * increase_gen_size
        self.settings.gen.line_width = gen_line_width

        # do the plot
        self.settings.gen.display_value = False
        self.settings.gen.display_name = False
        self.settings.sub.display_name= False
        self.settings.load.display_name = False
        
        self.settings.gen.cmap = self._convert_colors_to_cmap([COLOR_GEN[i] for i in range(len(TYPE_GEN))])
        gen_values = [TYPE_GEN[el] for el in self.observation_space.gen_type]
        self.figure = self.plot_info(gen_values=gen_values, coloring="gen")
        self.add_legend_gentype()

        # restore the state to its initial configuration
        self._restore_plot_charact(data)

        return self.figure

    def plot_current_dispatch(self, obs:BaseObservation, do_plot_actual_dispatch:bool=True,
                              increase_gen_size:float=1.5, gen_line_width:float=3,
                              palette_name:str="coolwarm"):
        # save the sate of the generators config
        data = self._save_plot_charact()

        # do the plot
        self.settings.sub.display_name = False
        self.settings.load.display_name = False
        self.assign_gen_palette(
            nb_color=5,
            color_scheme=palette_name,
            increase_gen_size=increase_gen_size,
            gen_line_width=gen_line_width,
        )
        if do_plot_actual_dispatch:
            gen_values = obs.actual_dispatch
        else:
            gen_values = obs.target_dispatch
        self.figure = self.plot_info(
            gen_values=gen_values, coloring="gen", gen_unit="MW"
        )

        # restore the state to its initial configuration
        self._restore_plot_charact(data)

        return self.figure

    def add_legend_gentype(self, loc:str="lower right"):
        """Add the legend for each generator type"""
        keys = sorted(TYPE_GEN.keys())
        ax_ = self.figure.axes[0]
        legend_help = [
            Line2D([0], [0], color=COLOR_GEN[TYPE_GEN[k]], label=k) for k in keys
        ]
        _ = ax_.legend(legend_help, keys, title="generator types", loc=loc)
        ax_.add_artist(self.legend)
