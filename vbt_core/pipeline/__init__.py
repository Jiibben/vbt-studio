"""Pipeline — orchestrates detection → tracking → analysis → rendering."""
from .runner import TrackResult, TrackingPipeline, run_tracking

__all__ = ["TrackingPipeline", "TrackResult", "run_tracking"]
