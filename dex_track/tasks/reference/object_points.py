# Ported from FlashDexRetarget 6f95e13 (MIT); see LICENSE.
import torch


def farthest_point_sample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Deterministic greedy farthest-point sampling over ``pts`` (V, 3): no RNG, the same
    ``n`` evenly-spread points are picked every call (stable obs). Returns (n,) index LongTensor."""
    V = pts.shape[0]
    if V <= n:
        # fewer verts than requested: use all, padding by repeating the last.
        pad = torch.full((n - V,), V - 1, dtype=torch.long, device=pts.device)
        return torch.cat([torch.arange(V, device=pts.device), pad])
    first = int(torch.argmax((pts - pts.mean(dim=0)).norm(dim=-1)))
    idx = torch.empty(n, dtype=torch.long, device=pts.device)
    idx[0] = first
    dist = (pts - pts[first]).norm(dim=-1)  # (V,) min dist to chosen set
    for i in range(1, n):
        nxt = int(torch.argmax(dist))
        idx[i] = nxt
        dist = torch.minimum(dist, (pts - pts[nxt]).norm(dim=-1))
    return idx
