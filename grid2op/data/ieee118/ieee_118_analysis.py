import pandapower as pp
import pandapower.topology as top
import pandapower.networks as nw
import pandapower.plotting as pplt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

palette = "muted"
n_colors = 8


# three different subgrids

def custom_legend(fig, entries, fontsize=8, loc='upper right', marker="o"):
    handles = list()

    for label, color in entries.items():
        handles.append(Line2D([0], [0], color=color, lw=2, label=label, marker=marker))

    fig.legend(handles=handles, fancybox=False,
               shadow=False, fontsize=fontsize, loc=loc, ncol=1, handletextpad=0.18)


def plot_feeder():
    net = nw.case118()
    fig, ax = plt.subplots(1, 1)
    mg = top.create_nxgraph(net, nogobuses=set(net.trafo.lv_bus.values))
    colors = sns.color_palette()
    collections = list()
    sizes = pplt.get_collection_sizes(net)
    voltage_levels = net.bus.vn_kv.unique()
    voltage_level_colors = dict(zip(voltage_levels, colors))
    legend_entries = dict()
    gens = set(net.gen.loc[:, "bus"].values)
    for area, color in zip(top.connected_components(mg), colors):
        vn_area = net.bus.loc[list(area)[0], "vn_kv"]
        color = voltage_level_colors[vn_area]
        legend_entries[vn_area] = color
        area_gens = gens - area
        other = area - gens

        collections.append(pplt.create_bus_collection(net, area_gens, color=color, size=sizes["bus"], zorder=11,
                                                      patch_type="rect"))
        collections.append(pplt.create_bus_collection(net, other, color=color, size=sizes["bus"], zorder=11))
        line_ind = net.line.loc[:, "from_bus"].isin(area) | net.line.loc[:, "to_bus"].isin(area)
        lines = net.line.loc[line_ind].index
        collections.append(pplt.create_line_collection(net, lines, color=color))

    eg_vn = net.bus.at[net.ext_grid.bus.values[0], "vn_kv"]
    collections.append(pplt.create_ext_grid_collection(net, size=sizes["ext_grid"], color=voltage_level_colors[eg_vn]))
    collections.append(pplt.create_trafo_collection(net, size=sizes["trafo"], zorder=1))
    pplt.draw_collections(collections, ax=ax)
    custom_legend(fig, entries=legend_entries)
    legend_entries = {"gen": "grey"}
    custom_legend(fig, entries=legend_entries, loc='center right', marker="s")
    print_info(net, fig)
    plt.show()


def print_info(net, fig):
    text = "Trafos: " + str(len(net.trafo)) + \
           "\nGens: " + str(len(net.gen)) + \
           "\nGen P [MW]: " + str(net.gen.p_mw.sum()) + \
           "\nLoads: " + str(len(net.load)) + \
           "\nLoad P [MW]: " + str(net.load.p_mw.sum()) + \
           "\nshunts: " + str(len(net.shunt)) + \
           "\nV levels: " + str(net.bus.vn_kv.unique())
    fig.text(0., 0., text)

if __name__ == "__main__":
    plot_feeder()
