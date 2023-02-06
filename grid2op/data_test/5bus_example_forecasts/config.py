from grid2op.Action import PowerlineSetAction
from grid2op.Reward import RedispReward
from grid2op.Rules import DefaultRules
from grid2op.Chronics import (Multifolder,
                              GridStateFromFileWithForecasts)

# TODO change this !
from grid2op.Backend import PandaPowerBackend
from grid2op.Opponent import BaseActionBudget

from grid2op.l2rpn_utils import ActionWCCI2022, ObservationWCCI2022

opponent_attack_cooldown = 12  # 1 hour, 1 hour being 12 time steps
opponent_attack_duration = 96  # 8 hours at maximum
opponent_budget_per_ts = (
    0.17  # opponent_attack_duration / opponent_attack_cooldown + epsilon
)
opponent_init_budget = 144.0  # no need to attack straightfully, it can attack starting at midday the first day

config = {
    "backend": PandaPowerBackend,
    "action_class": ActionWCCI2022,
    "observation_class": ObservationWCCI2022,
    "reward_class": RedispReward,
    "gamerules_class": DefaultRules,
    # TODO change that too
    "chronics_class": Multifolder,
    "grid_value_class": GridStateFromFileWithForecasts,
    "data_feeding_kwargs": {"h_forecast": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]},
    "volagecontroler_class": None,
    "names_chronics_to_grid": None,
    "opponent_attack_cooldown": opponent_attack_cooldown,
    "opponent_attack_duration": opponent_attack_duration,
    "opponent_budget_per_ts": opponent_budget_per_ts,
    "opponent_init_budget": opponent_init_budget,
    "opponent_action_class": PowerlineSetAction,
    "opponent_budget_class": BaseActionBudget,
}
