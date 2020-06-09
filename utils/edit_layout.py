#!/usr/bin/env python3

import sys
import os
import json
import argparse

import grid2op
from grid2op.PlotGrid import PlotMatplot

def edit_layout(ds_name, test=False):
    env = grid2op.make(ds_name, test=test)

    plotter = PlotMatplot(env.observation_space)
    fig = plotter.plot_layout()
    fig.show()

    user_input = ""
    while True:
        # Select a substation or exit
        user_input = input("exit or sub id: ")
        if "exit" in user_input:
            break
        sub_id = int(user_input)

        # Get substation infos
        sub_name = env.name_sub[sub_id]
        x = plotter._grid_layout[sub_name][0]
        y = plotter._grid_layout[sub_name][1]
        print ("{} [{};{}]".format(sub_name, x, y))

        # Update x coord
        user_input = input("{} new x: ".format(sub_name))
        if len(user_input) == 0:
            new_x = x
        else:
            new_x = float(user_input)

        # Update Y coord
        user_input = input("{} new y: ".format(sub_name))
        if len(user_input) == 0:
            new_y = y
        else:
            new_y = float(user_input)

        # Apply to layout
        plotter._grid_layout[sub_name][0] = new_x
        plotter._grid_layout[sub_name][1] = new_y

        # Redraw
        plotter.plot_info(figure=fig)
        fig.canvas.draw()

    # Done editing, print subs result
    subs_layout = {}
    for k, v in plotter._grid_layout.items():
        if k in env.name_sub:
            subs_layout[k] = v
    print(json.dumps(subs_layout, indent=2))
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid layout editor')
    parser.add_argument('--dataset',
                        required=True, type=str,                        
                        help='Path to dataset directory')
    parser.add_argument('--test',
                        default=False, action="store_true",
                        help='Pass test=True to grid2op.make')
    args = parser.parse_args()    
    edit_layout(args.dataset, test=args.test)
