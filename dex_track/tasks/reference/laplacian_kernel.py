"""GPU on-the-fly reference weights and L*(actual-reference), no intermediates."""
import torch
import triton
import triton.language as tl


@triton.jit
def _laplacian_rows(reference, delta, adjacency, output, N: tl.constexpr = 71,
                    STRIDE: tl.constexpr = 9, BLOCK: tl.constexpr = 128, HAND_OBJECT: tl.constexpr = False):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < N
    bits = tl.load(adjacency+(batch*N+row)*STRIDE+col//8, valid, other=0).to(tl.int32)
    connected = valid & ((bits >> (col%8)) & 1 != 0)
    if HAND_OBJECT:
        connected = connected & (col >= 21)
    i = (batch*N+row)*3
    j = (batch*N+col)*3
    xi = tl.load(reference+i); yi = tl.load(reference+i+1); zi = tl.load(reference+i+2)
    xj = tl.load(reference+j, valid, other=0)
    yj = tl.load(reference+j+1, valid, other=0)
    zj = tl.load(reference+j+2, valid, other=0)
    distance = tl.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj))
    weight = tl.where(connected, tl.exp(-30.*distance), 0.)
    total = tl.maximum(tl.sum(weight, 0), 1.e-20)
    dx = tl.load(delta+i)-tl.sum(weight*tl.load(delta+j, valid, other=0), 0)/total
    dy = tl.load(delta+i+1)-tl.sum(weight*tl.load(delta+j+1, valid, other=0), 0)/total
    dz = tl.load(delta+i+2)-tl.sum(weight*tl.load(delta+j+2, valid, other=0), 0)/total
    value = dx*dx+dy*dy+dz*dz
    if HAND_OBJECT:
        value = tl.where(tl.sum(weight, 0) > 0., value, 0.)
    tl.store(output+batch*N+row, value)


def laplacian_on_the_fly(reference, actual, adjacency, graph_mode="full"):
    if graph_mode not in ("full", "hand_object"):
        raise ValueError(graph_mode)
    rows = 21 if graph_mode == "hand_object" else 71
    """71 vertices, adjacency packed into nine uint8 bytes per source row."""
    delta = (actual-reference).contiguous()
    reference = reference.contiguous()
    adjacency = adjacency.contiguous()
    if reference.is_cuda:
        squared = torch.empty(reference.shape[:-1], device=reference.device, dtype=reference.dtype)
        _laplacian_rows[(reference.numel()//(71*3), rows)](
            reference, delta, adjacency, squared, HAND_OBJECT=graph_mode == "hand_object", num_warps=4)
    else:
        columns = torch.arange(71, device=reference.device)
        mask = ((adjacency[..., columns//8].to(torch.int32) >> (columns%8)) & 1).bool()
        if graph_mode == "hand_object":
            mask = mask & (columns >= 21)
        distance = (reference[..., :, None, :]-reference[..., None, :, :]).norm(dim=-1)
        weights = torch.exp(-30*distance)*mask
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-20)
        squared = (delta-torch.matmul(weights, delta)).square().sum(-1)
        if graph_mode == "hand_object":
            squared = squared.masked_fill(~mask.any(-1), 0.)
    return squared[..., :rows].mean(-1).sqrt()
