"""Consecutive exceed counter, reset on recovery as in mimic-lite."""
import torch

class ConsecutiveExceed:
    def __init__(self, shape, device, threshold=.2, min_steps=5):
        self.threshold, self.min_steps = threshold, min_steps
        self.steps = torch.zeros(shape, device=device, dtype=torch.int32)

    def reset(self, env_ids):
        self.steps[env_ids] = 0

    def update(self, error, mask):
        self.steps = torch.where(mask & (error > self.threshold), self.steps + 1, 0)
        return self.steps >= self.min_steps
