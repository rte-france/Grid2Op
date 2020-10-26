__all__ = [
    "RulesChecker",
    "DefaultRules",
    "AlwaysLegal",
    "BaseRules",
    "LookParam",
    "PreventReconnection",

]

from grid2op.Rules.RulesChecker import RulesChecker
from grid2op.Rules.DefaultRules import DefaultRules
from grid2op.Rules.AlwaysLegal import AlwaysLegal
from grid2op.Rules.BaseRules import BaseRules
from grid2op.Rules.LookParam import LookParam
from grid2op.Rules.PreventReconnection import PreventReconnection
import warnings


class LegalAction(BaseRules):
    def __init__(self, *args, **kwargs):
        BaseRules.__init__(self, *args, **kwargs)
        warnings.warn("LegalAction class has been renamed \"BaseRules\". "
                      "This class LegalAction will be removed in future versions.",
                      category=PendingDeprecationWarning)


class GameRules(RulesChecker):
    def __init__(self, *args, **kwargs):
        RulesChecker.__init__(self, *args, **kwargs)
        warnings.warn("GameRules class has been renamed \"RulesChecker\". "
                      "This class GameRules will be removed in future versions.",
                      category=PendingDeprecationWarning)


class PreventReconection(PreventReconnection):
    def __init__(self, *args, **kwargs):
        PreventReconection.__init__(self, *args, **kwargs)
        warnings.warn("PreventReconection class has been renamed \"PreventReconnection\". "
                      "This class Action will be removed in future versions.",
                      category=PendingDeprecationWarning)
