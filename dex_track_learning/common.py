"""PPO helpers local to dex-track; no mimic-lite package import."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY
import active_adaptation as aa

OBJECT_PCD_KEY = "object_pcd"
OBJECT_PCD_FEATURE_KEY = "_object_pcd_feature"


class BimanualPointCloudEncoder(nn.Module):
    """Shared point MLP, separate wrist-frame pooling in right/left hand order."""

    def __init__(self, num_points=128, hidden_dims=(64, 128), input_is_scene=False):
        super().__init__()
        self.num_points = int(num_points)
        self.input_is_scene = bool(input_is_scene)
        if self.num_points < 1 or len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("Expected positive point count and two positive hidden widths")
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dims[0]), nn.Mish(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Mish(),
        )

    def compile_pool(self):
        # Keep checkpoint/optimizer parameter names stable and avoid CUDA graph pools.
        self._pool = torch.compile(
            self._pool, fullgraph=True, dynamic=True,
            mode="max-autotune-no-cudagraphs")

    def forward(self, points):
        if self.input_is_scene:
            points = points[..., -(2*self.num_points*3):]
        if points.shape[-1] != 2 * self.num_points * 3:
            raise ValueError("Expected two flattened wrist-frame point clouds")
        prefix = points.shape[:-1]
        points = points.reshape(-1, 2, self.num_points, 3)
        # Recompute large PPO point activations in bounded chunks during backward.
        # Samples remain independent; the optimizer still sees the full minibatch.
        if torch.is_grad_enabled() and points.shape[0] > 1024:
            features = torch.cat([
                checkpoint(self._pool, chunk, use_reentrant=False)
                for chunk in points.split(1024)
            ], dim=0)
        else:
            features = self._pool(points)
        return features.reshape(*prefix, features.shape[-1])

    def _pool(self, points):
        # Preserve wrist-relative translation and hand identity; pool only points.
        return self.point_mlp(points).max(dim=-2).values.flatten(-2)


class NullVecNorm(VecNorm):
    """Identity VecNorm that keeps the module/state interface intact."""

    def forward(self, input_vector: torch.Tensor):
        return input_vector

    def _update(self, input_vector: torch.Tensor):
        raise RuntimeError("NullVecNorm does not support updating statistics.")

    def _compute(self):
        raise RuntimeError("NullVecNorm does not compute normalization.")

    def synchronize(self, mode: str = "broadcast"):
        del mode
        return None

def check_vecnorm_divergence(vecnorm: VecNorm):
    world_size = aa.get_world_size()

    loc_scale = torch.stack(vecnorm._compute())
    gathered = [torch.empty_like(loc_scale) for _ in range(world_size)]
    dist.all_gather(gathered, loc_scale)

    loc_diffs = []
    scale_diffs = []
    for i in range(world_size):
        loc_diff = torch.abs(gathered[i][0] - loc_scale[0]).sum().item()
        scale_diff = torch.abs(gathered[i][1] - loc_scale[1]).sum().item()
        loc_diffs.append(loc_diff)
        scale_diffs.append(scale_diff)
    return loc_diffs, scale_diffs

class MeanAction(TensorDictModuleBase):
    in_keys = ["loc"]
    out_keys = [ACTION_KEY]

    def forward(self, td):
        td[ACTION_KEY] = td["loc"]
        return td

class ObsOODDetector(TensorDictModuleBase):
    def __init__(self, in_keys, sigma: float = 5.0):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = [("next", f"{k}_ood_ratio") for k in in_keys] + [
            ("next", k) for k in in_keys
        ]
        self.sigma = sigma

    def forward(self, tensordict: TensorDict):
        for in_key in self.in_keys:
            obs = tensordict.get(in_key, None)
            if obs is not None:
                ood_ratio = (obs.abs() > self.sigma).float().mean(dim=-1, keepdim=True)
                tensordict.set(("next", f"{in_key}_ood_ratio"), ood_ratio)
                tensordict.set(("next", in_key), obs)
        return tensordict

class ActorROA(nn.Module):
    def __init__(
        self,
        action_dim: int,
        init_noise_scale: float = 1.0,
        load_noise_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * init_noise_scale)
        self.scale_mapping = nn.Identity()
        self.load_noise_scale = load_noise_scale

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.ones_like(loc) * self.actor_std
        scale = self.scale_mapping(scale)
        return loc, scale

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.load_noise_scale is not None:
            self.actor_std.data.fill_(self.load_noise_scale)


class PreserveScenePointCloud(nn.Module):
    """Normalize link features only; raw xyz tail stays in physical metres."""
    def __init__(self, norm, point_width):
        super().__init__()
        self.norm, self.point_width = norm, point_width

    def forward(self, scene):
        return torch.cat((self.norm(scene[..., :-self.point_width]),
                          scene[..., -self.point_width:]), -1)


class SceneCatTensors(TensorDictModuleBase):
    """Heads consume scene link features and encoded points, never raw xyz twice."""
    def __init__(self, in_keys, out_key, point_width):
        super().__init__()
        self.in_keys, self.out_keys = list(in_keys), [out_key]
        self.point_width = point_width

    def forward(self, td):
        parts = [td[k][..., :-self.point_width] if k == 'scene' else td[k] for k in self.in_keys]
        return td.set(self.out_keys[0], torch.cat(parts, -1))


class ActionRouting(TensorDictModuleBase):
    def __init__(self, action_spec):
        super().__init__()
        self.keys = list(action_spec.keys())
        if "action" in self.keys and len(self.keys) != 1:
            raise ValueError("Multi-input PPO reserves 'action' for the flat sampled action")
        self.sizes = [action_spec[key].shape[-1] for key in self.keys]
        self.action_dim = sum(self.sizes)
        self.in_keys = ["action"]
        self.out_keys = self.keys

    def forward(self, tensordict):
        for key, value in zip(self.keys, torch.split(tensordict["action"], self.sizes, dim=-1)):
            tensordict.set(key, value)
        return tensordict
