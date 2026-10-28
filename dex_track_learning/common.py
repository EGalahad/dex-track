"""PPO helpers local to dex-track; no mimic-lite package import."""

import torch
import torch.distributed as dist
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY
import active_adaptation as aa

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
