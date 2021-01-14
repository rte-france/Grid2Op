import pandapower as pp
import pandapower.networks as pn
import copy

case14_modif = pn.case14()
# add a bus
id_bus_created = pp.create_bus(case14_modif, vn_kv=135., name="bus_added_for_test")

# add a line that go to this bus (copy of the powerline 0)
ref_line = dict(copy.deepcopy(case14_modif.line.iloc[0]))
del ref_line["name"]
del ref_line["from_bus"]
del ref_line["to_bus"]
id_line_added = pp.create_line_from_parameters(case14_modif, from_bus=0, to_bus=id_bus_created,
                                               name="line_added_for_test",
                                               **ref_line)

# add a storage unit to this bus
id_bus_created = pp.create_storage(case14_modif, bus=id_bus_created, p_mw=20., max_e_mwh=20.,
                                   name="storage_added_for_test")

# the powerflow runs correctly
pp.runpp(case14_modif)
assert case14_modif.converged, "the grid as converged"
assert case14_modif.res_storage["p_mw"].iloc[0] == 20.

# now i switch off the created powerline (so that the storage unit is not connected to the grid
case14_modif.line["in_service"].values[id_line_added] = False
assert not case14_modif.line.iloc[id_line_added]["in_service"]

# i start a new powerflow
pp.runpp(case14_modif, check_connectivity=False)
assert case14_modif.converged, "the grid as converged"
print(case14_modif.res_storage["p_mw"].iloc[0])

