"""Local object samples used only by the tracking reward and termination."""
import math


def normalize_tracking_points(points, radius):
    """Set max distance from the object origin to ``radius`` metres."""
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Object tracking radius must be finite and positive')
    return points * (radius / points.norm(dim=-1).max().clamp_min(1e-6))
