from grid2op.Action import BaseAction

class CustomAction(BaseAction):
    def __init__(self, gridobj,
                 setSubset=True,
                 changeSubset=True,
                 redispatchSubset=True):
        super().__init__(gridobj)

        self.attr_list_vect = []
        if setSubset:
            self.attr_list_vect.append("_set_line_status")
            self.attr_list_vect.append("_set_topo_vect")
        if changeSubset:
            self.attr_list_vect.append("_change_bus_vect")
            self.attr_list_vect.append("_switch_line_status")
        if redispatchSubset:
            self.attr_list_vect.append("_redispatch")

    def __call__(self):
        return super().__call__()
