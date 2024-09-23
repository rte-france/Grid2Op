__all__ = ["EpisodeData"]

from grid2op.Episode.EpisodeData import EpisodeData
from grid2op.Episode.CompactEpisodeData import CompactEpisodeData

# Try to import optional module
try:
    from grid2op.Episode.EpisodeReplay import EpisodeReplay

    __all__.append("EpisodeReplay")
except ImportError:
    pass  # Silent fail for optional dependencies
