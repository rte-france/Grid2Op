import matplotlib.pyplot as plt
import pandas as pd

import pandapower as pp
import pandapower.networks as nw
import pandapower.plotting as pplt


def get_net_118_with_zones():
    net = nw.case118()
    pp.runpp(net)
    net.bus.sort_index(inplace=True)

    net.bus.loc[:32, "zone"] = 1
    net.bus.loc[32:67, "zone"] = 2
    net.bus.loc[67:112, "zone"] = 3
    net.bus.loc[23, "zone"] = 3
    net.bus.loc[[112, 113, 114, 115, 116], "zone"] = 1
    net.bus.loc[[115, 117], "zone"] = 3
    net.bus.loc[:, "toy_zone"] = False
    net.bus.loc[99:112, "toy_zone"] = True
    net.bus.loc[[100, 101], "toy_zone"] = False
    return net


def get_subnets(net):
    areas = dict()
    for zone in net.bus.zone.unique():
        zone_buses = net.bus.loc[net.bus.zone == zone].index
        areas[zone] = pp.select_subnet(net, zone_buses)
    return areas


def plot_zones():
    net = get_net_118_with_zones()
    areas = get_subnets(net)

    fig, axes = plt.subplots(1, len(areas.keys()))

    keys = areas.keys()
    keys = sorted(keys)
    for i, zone in enumerate(keys):
        net = areas[zone]
        collections = list()
        ax = axes[i]
        sizes = pplt.get_collection_sizes(net)
        collections.append(pplt.create_bus_collection(net, size=sizes["bus"]))
        collections.append(pplt.create_line_collection(net))
        collections.append(pplt.create_trafo_collection(net, size=sizes["trafo"]))
        if zone == 3:
            collections.append(pplt.create_bus_collection(net, net.bus.loc[net.bus.toy_zone].index,
                                                          color="g", size=2 * sizes["bus"], zorder=11))
        pplt.draw_collections(collections, ax=ax)

    plt.show()


def create_toy_zone():
    net = nw.case118()
    pp.runpp(net)
    vm_ext_grid = net.res_bus.loc[99, "vm_pu"]
    va_ext_grid = net.res_bus.loc[99, "va_degree"]
    net = get_net_118_with_zones()
    areas = get_subnets(net)

    net = areas[3]
    net = pp.select_subnet(net, buses=net.bus.loc[net.bus.toy_zone].index)

    pp.create_ext_grid(net, bus=99, vm_pu=vm_ext_grid, va_degree=va_ext_grid)
    pp.runpp(net)

    return net


def create_zone3():
    net118 = nw.case118()
    pp.runpp(net118)
    vm_gen = net118.res_bus.at[23, "vm_pu"]

    p_mw_gen = net118.res_line.loc[net118.line.from_bus.isin([23, 24]) & net118.line.to_bus.isin([23, 24]), "p_from_mw"]
    p_mw_gen = abs(sum(p_mw_gen))

    net = get_net_118_with_zones()
    areas = get_subnets(net)
    net = areas[3]

    pp.create_gen(net, bus=23, vm_pu=vm_gen, p_mw=p_mw_gen)

    pp.runpp(net)
    return net


def create_zone1():
    net118 = nw.case118()
    pp.runpp(net118)
    vm_ext_grid = net118.res_bus.loc[[32, 33, 37], "vm_pu"].mean()
    va_ext_grid = net118.res_bus.loc[[32, 33, 37], "va_degree"].mean()
    vm_gen = net118.res_bus.at[22, "vm_pu"]

    p_mw_gen = net118.res_line.loc[net118.line.from_bus.isin([22, 23]) & net118.line.to_bus.isin([22, 23]), "p_from_mw"]
    p_mw_gen = abs(sum(p_mw_gen))

    net = get_net_118_with_zones()
    areas = get_subnets(net)
    net = areas[1]

    zone1_lines = [49, 41, 40]

    b = pp.create_bus(net, vn_kv=net118.bus.loc[33, "vn_kv"], name="zone2_slack", index=118,
                      geodata=(net118.bus_geodata.at[33, "x"], net118.bus_geodata.at[33, "y"]))

    net.line = pd.concat([net.line, net118.line.loc[zone1_lines]], sort=False)
    net.line.loc[zone1_lines, "to_bus"] = int(b)

    pp.create_ext_grid(net, bus=b, vm_pu=vm_ext_grid, va_degree=va_ext_grid)

    pp.create_gen(net, bus=22, vm_pu=vm_gen, p_mw=p_mw_gen)

    pp.runpp(net)
    return net


if __name__ == "__main__":
    plot_zones()
    toy_net = create_toy_zone()
    net_zone3 = create_zone3()
    net_zone1 = create_zone1()
