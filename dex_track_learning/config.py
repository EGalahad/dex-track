"""Validate the supported shared PointNet policy."""
def normalize_policy_config(cfg):
    if not cfg.get('use_pointnet', False):
        raise ValueError('dex-track requires the shared PointNet policy.')
    return cfg
