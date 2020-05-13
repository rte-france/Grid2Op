__all__ = [
    "EpisodeData"
]

from grid2op.Episode.EpisodeData import EpisodeData

# Try to import optional module
try:
    from grid2op.Episode.EpisodeReplay import EpisodeReplay
    __all__.append("EpisodeReplay")
except ImportError:
    pass  # Silent fail for optional dependencies
