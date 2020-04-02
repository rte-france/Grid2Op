from grid2op.Action import TopoAndRedispAction

class CustomAction(TopoAndRedispAction):
    def __init__(self, gridobj):
        super().__init__(gridobj)

        # Only use "set" actions
        self.attr_list_vect = ["_set_line_status",
                               "_set_topo_vect" ]

    def __call__(self):
        return super().__call__()
