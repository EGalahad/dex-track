"""Experiment run helpers for dex-track train.py (Trackio local + optional W&B).

Kept inside the project so AA can stay on ``dev/dex-track`` until the logger
merge. Trackio defaults to a local SQLite store (no Hugging Face Space).
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _ConfigView:
    def __init__(self, run: "ExperimentRun") -> None:
        self._run = run
        self._pending: dict[str, Any] = {}

    def update(self, data: Mapping[str, Any]) -> None:
        payload = _json_ready(dict(data))
        raw = self._run._raw
        if raw is None:
            self._pending.update(payload)
            return
        if self._run.backend == "trackio":
            cfg = getattr(raw, "config", None)
            if isinstance(cfg, dict):
                cfg.update(payload)
            elif cfg is not None and hasattr(cfg, "update"):
                cfg.update(payload)
            else:
                self._pending.update(payload)
            return
        raw.config.update(payload)

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})


class ExperimentRun:
    """Small wandb-shaped facade: ``name``, ``id``, ``dir``, ``config``, ``log``, ``save``, ``finish``."""

    def __init__(self, *, backend: str, project: str, mode: str, job_type: str | None,
                 tags: list[str] | None, space_id: str | None = None,
                 run_id: str | None = None) -> None:
        if backend not in ("trackio", "wandb"):
            raise ValueError(f"Unsupported logger backend {backend!r}; use trackio or wandb")
        self.backend = backend
        self.project = project
        self.mode = mode
        self.job_type = job_type
        self.tags = list(tags or [])
        self.space_id = space_id
        self._id = (run_id or uuid.uuid4().hex)[:32]
        self._name = f"run-{self._id[:8]}"
        self._files_dir: Path | None = None
        self._raw: Any = None
        self.config = _ConfigView(self)
        if mode == "disabled":
            self._files_dir = self._make_files_dir(self._id)
            self._raw = _DisabledRun(self._id, self._name)

    @property
    def id(self) -> str:
        if self._raw is not None and getattr(self._raw, "id", None) is not None:
            return str(self._raw.id)
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)
        if self._raw is None or self.mode == "disabled":
            if self._raw is not None:
                self._raw.name = self._name
            return
        if self.backend == "wandb":
            self._raw.name = self._name
            return
        # Trackio fixes the name at init; rename is best-effort for local bookkeeping.
        _log.info("Trackio run name is fixed at init (%s); local label=%s",
                  getattr(self._raw, "name", None), self._name)

    @property
    def dir(self) -> str:
        self._ensure()
        assert self._files_dir is not None
        return str(self._files_dir)

    def log(self, data: Mapping[str, Any], step: int | None = None) -> None:
        self._ensure()
        if self.mode == "disabled":
            return
        payload = dict(data)
        if self.backend == "trackio":
            self._raw.log(payload, step=step)
            return
        self._raw.log(payload, step=step)

    def save(self, path: str, policy: str = "now", base_path: str | None = None) -> None:
        del policy, base_path
        self._ensure()
        if self.mode == "disabled":
            return
        src = Path(path)
        if not src.exists():
            return
        if self.backend == "wandb":
            self._raw.save(str(src), policy="now", base_path=base_path or self.dir)
            return
        # Trackio has no wandb-style file sync; keep a copy under the run files dir.
        dest_dir = Path(self.dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    def finish(self) -> None:
        if self._raw is None or self.mode == "disabled":
            return
        if self.backend == "trackio":
            import trackio
            trackio.finish()
            return
        self._raw.finish()

    def _ensure(self) -> None:
        if self._raw is not None:
            return
        if self.backend == "trackio":
            self._start_trackio()
        else:
            self._start_wandb()

    def _start_trackio(self) -> None:
        import trackio

        pending = dict(self.config._pending)
        if self.job_type:
            pending.setdefault("job_type", str(self.job_type))
        if self.tags:
            pending.setdefault("tags", list(self.tags))

        kwargs: dict[str, Any] = {
            "project": self.project,
            "name": self._name.replace("/", "-"),
            "config": pending or None,
            "embed": False,
            "auto_log_gpu": False,
            "auto_log_cpu": False,
        }
        want_cloud = self.mode == "online"
        if want_cloud and self.space_id:
            kwargs["space_id"] = str(self.space_id)
        elif want_cloud:
            _log.info("Trackio mode=online without space_id; logging locally.")
            want_cloud = False

        hub_prev = os.environ.get("HF_HUB_OFFLINE")
        if not want_cloud:
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._raw = trackio.init(**kwargs)
        finally:
            if not want_cloud:
                if hub_prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = hub_prev
        self.config._pending.clear()
        self._id = str(self._raw.id)
        self._files_dir = self._make_files_dir(self._id)
        print(f"[trackio] project={self.project} run={self._raw.name} id={self._id}", flush=True)
        print(f"[trackio] files={self._files_dir}", flush=True)
        print(f'[trackio] dashboard: trackio show --project "{self.project}"', flush=True)

    def _start_wandb(self) -> None:
        import wandb

        self._raw = wandb.init(
            job_type=self.job_type,
            project=self.project,
            mode=self.mode,
            tags=self.tags,
            id=self._id if len(self._id) >= 8 else None,
        )
        if self.config._pending:
            self._raw.config.update(self.config._pending)
            self.config._pending.clear()
        self._id = str(self._raw.id)
        self._files_dir = Path(self._raw.dir)
        self._files_dir.mkdir(parents=True, exist_ok=True)

    def _make_files_dir(self, run_id: str) -> Path:
        # Mirror W&B: keep cfg/checkpoints under the Hydra job cwd
        # (``outputs/<date>/<run>/trackio/offline-run-.../files``), not the
        # global Trackio SQLite cache.
        import datetime as _dt
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        short = str(run_id)[:8]
        files = Path.cwd() / "trackio" / f"offline-run-{stamp}-{short}" / "files"
        files.mkdir(parents=True, exist_ok=True)
        return files


class _DisabledRun:
    def __init__(self, run_id: str, name: str) -> None:
        self.id = run_id
        self.name = name
        self.config: dict[str, Any] = {}

    def log(self, *args: Any, **kwargs: Any) -> None:
        return

    def finish(self) -> None:
        return

    def save(self, *args: Any, **kwargs: Any) -> None:
        return


def init_experiment_run(cfg) -> ExperimentRun:
    """Build a run from ``cfg.logger`` (preferred) or legacy ``cfg.wandb``."""
    logger_cfg = cfg.get("logger")
    if logger_cfg is None:
        wb = cfg.wandb
        return ExperimentRun(
            backend="wandb",
            project=str(wb.project),
            mode=str(wb.mode),
            job_type=str(wb.get("job_type", "train")),
            tags=list(wb.get("tags", []) or []),
            run_id=wb.get("id", None),
        )
    backend = str(logger_cfg.get("backend", "trackio"))
    return ExperimentRun(
        backend=backend,
        project=str(logger_cfg.project),
        mode=str(logger_cfg.get("mode", "offline")),
        job_type=str(logger_cfg.get("job_type", "train")),
        tags=list(logger_cfg.get("tags", []) or []),
        space_id=logger_cfg.get("space_id", None),
        run_id=logger_cfg.get("id", None),
    )


def upload_checkpoints_enabled(cfg) -> bool:
    logger_cfg = cfg.get("logger")
    if logger_cfg is not None and "upload_checkpoints" in logger_cfg:
        return bool(logger_cfg.upload_checkpoints)
    return bool(cfg.wandb.get("upload_checkpoints", True))
