#!/usr/bin/env python3

import os
import argparse
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html

from grid2op.Action import ActionSpace
from grid2op.Observation import ObservationSpace
from grid2op.Plot.PlotPlotly import PlotPlotly

def cli():
    parser = argparse.ArgumentParser(description="Graph inspector")
    parser.add_argument("--logdir", required=True,
                        help="Path to the runner output directory")
    parser.add_argument("--episode", required=False,
                        default="000", type=str,
                        help="Name of the episode to inspect")
    return parser.parse_args()

class VizServer:
    def __init__(self, args):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.episode = self.load(args)
        self.app.layout = self.setupLayout()
        self.plot_helper = PlotPlotly(observation_space=self.episode["obs_space"])

        # Register slider update callback
        self.app.callback([dash.dependencies.Output("grid-graph", "figure")],
                          [dash.dependencies.Input("step-slider", "value")])(self.updateGraph)
        # Register action update callback
        self.app.callback([dash.dependencies.Output("last-action", "children"),
                           dash.dependencies.Output("next-action", "children")],
                          [dash.dependencies.Input("step-slider", "value")])(self.updateAction)
        # Register buttons update callback
        self.app.callback([dash.dependencies.Output("step-slider", "value")],
                          [dash.dependencies.Input("prev-step", "n_clicks"),
                           dash.dependencies.Input("next-step", "n_clicks")],
                          [dash.dependencies.State("step-slider", "value"),
                           dash.dependencies.State("step-slider", "min"),
                           dash.dependencies.State("step-slider", "max")])(self.triggerButtons)

    @staticmethod
    def load(args):
        ACT_JSON = "dict_action_space.json"
        ACT_DATA = "actions.npy"
        ENV_JSON = "dict_env_modification_space.json"
        ENV_DATA = "env_modifications.npy"
        OBS_JSON = "dict_observation_space.json"
        OBS_DATA = "observations.npy"

        act_json_abs = os.path.abspath(os.path.join(args.logdir, ACT_JSON))
        act_data_abs = os.path.abspath(os.path.join(args.logdir, args.episode, ACT_DATA))
        env_json_abs = os.path.abspath(os.path.join(args.logdir, ENV_JSON))
        env_data_abs = os.path.abspath(os.path.join(args.logdir, args.episode, ENV_DATA))
        obs_json_abs = os.path.abspath(os.path.join(args.logdir, OBS_JSON))
        obs_data_abs = os.path.abspath(os.path.join(args.logdir, args.episode, OBS_DATA))

        act_space = ActionSpace.from_dict(act_json_abs)
        act_data = np.load(act_data_abs)
        env_space = ActionSpace.from_dict(env_json_abs)
        env_data = np.load(env_data_abs)
        obs_space = ObservationSpace.from_dict(obs_json_abs)
        obs_data = np.load(obs_data_abs)

        # Filter out steps that did not run (containing NaNs)
        obs_steps = np.array(list(range(obs_data.shape[0])))
        valid_obs_steps = obs_steps[np.all(np.logical_not(np.isnan(obs_data)), axis=1)]
        episode = {
            "size": valid_obs_steps.shape[0],
            "act_space": act_space,
            "env_space": env_space,
            "obs_space": obs_space,
            "steps": []
        }
        for step in range(episode["size"] - 1):
            step_dict = {
                "act": act_space.from_vect(act_data[step]),
                "env": None, #env_space.from_vect(env_data[step]),
                "obs": obs_space.from_vect(obs_data[step])
            }
            episode["steps"].append(step_dict)

        return episode

    def setupLayout(self):
        title = html.H1(children='Viz demo')
        graph = dcc.Graph(id="grid-graph")
        self.prev_clicks = 0
        prev_step_button = html.Button("Prev", id="prev-step", n_clicks=0)
        self.next_clicks = 0
        next_step_button = html.Button("Next", id="next-step", n_clicks=0)
        self.slider = dcc.Slider(
            id="step-slider",
            min=0,
            max=self.episode["size"] - 2,
            value=0)
        last_action = html.Div(id="last-action")
        next_action = html.Div(id="next-action")

        body = [
            title, graph,
            prev_step_button,
            next_step_button,
            self.slider,
            last_action, next_action
        ]
        layout = html.Div(children=body)
        return layout

    def updateGraph(self, step):
        step_obs = self.episode["steps"][step]["obs"]
        fig = self.plot_helper.get_plot_observation(step_obs)
        fig.update_layout(
            height=800,
            width=800
        )
        return [fig]

    def updateAction(self, step):
        prev_step = step if step >= 0 and step < self.episode["size"] else 0
        prev_act = self.episode["steps"][prev_step]["act"]
        curr_act = self.episode["steps"][step]["act"]
        if prev_step != 0:
            html_prev_act = [html.P("Previous action:"), html.P(str(prev_act))]
        else:
            html_prev_act = [html.P("Previous action:"), html.P("N/A")]
        html_curr_act = [html.P("Next Action:"), html.P(str(curr_act))]
        return [html_prev_act, html_curr_act]

    def triggerButtons(self, prev_clicks, next_clicks, slider_value, slider_min, slider_max):
        new_value = 0
        if self.prev_clicks < prev_clicks:
            self.prev_clicks = prev_clicks
            new_value = slider_value - 1

        if self.next_clicks < next_clicks:
            self.next_clicks = next_clicks
            new_value = slider_value + 1

        new_value = np.clip(new_value, slider_min, slider_max)
        return [new_value]

    def run(self, debug=False):
        self.app.run_server(debug=debug)

if __name__ == '__main__':
    args = cli()
    server = VizServer(args)
    server.run()
