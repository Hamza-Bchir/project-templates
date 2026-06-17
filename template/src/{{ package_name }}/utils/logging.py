import warnings
from typing import Any, Literal, Sequence

import torch

DEFAULT_STEP_METRICS: dict[str, str] = {
    "train/": "train/step",
    "train_epoch/": "train_epoch/step",
    "val/": "val/step",
}


class WandbLogger:
    """
    Wrapper class for wandb methods.
    See : https://docs.wandb.ai/models/track/log/customize-logging-axes
    for customized log axes.
    """

    def __init__(
        self,
        *,
        entity: str | None,
        project: str,
        dir: str | None,
        name: str,
        id: str | None = None,
        notes: str | None = None,
        tags: Sequence[str] | None = None,
        config: dict[str, Any] | str | None = None,
        config_exclude_keys: list[str] | None = None,
        config_include_keys: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        mode: Literal["online", "offline", "disabled", "shared"] | None = None,
        reinit: bool
        | Literal["default", "return_previous", "finish_previous", "create_new"]
        | None = "finish_previous",
        resume: bool | Literal["allow", "never", "must", "auto"] | None = None,
        step_metrics: dict[str, str] | None = None,
        warn_missing_step: bool = True,
        do_watch: bool = False,
        watch_log: Literal["gradients", "parameters", "all"] | None = "all",
        watch_log_freq: int = 1000,
    ) -> None:

        self.entity = entity
        self.project = project
        self.dir = dir
        self.name = name
        self.id = id
        self.notes = notes
        self.tags = list(tags) if tags is not None else []
        self.config = config
        self.config_exclude_keys = config_exclude_keys
        self.config_include_keys = config_include_keys
        self.group = group
        self.job_type = job_type
        self.mode = mode
        self.reinit = reinit
        self.resume = resume
        self.step_metrics = dict(DEFAULT_STEP_METRICS) if step_metrics is None else dict(step_metrics)
        self._run = None
        self.warn_missing_step = warn_missing_step
        self._warned_prefixes: set[str] = set()
        self.do_watch = do_watch
        self.watch_log = watch_log
        self.watch_log_freq = watch_log_freq

    def init(self, config: dict[str, Any] | None = None) -> None:
        import wandb

        self._run = wandb.init(
            entity=self.entity,
            project=self.project,
            dir=self.dir,
            name=self.name,
            id=self.id,
            notes=self.notes,
            tags=self.tags,
            config={"config": config} if config is not None else {"config": self.config},
            config_exclude_keys=self.config_exclude_keys,
            config_include_keys=self.config_include_keys,
            group=self.group,
            job_type=self.job_type,
            mode=self.mode,
            reinit=self.reinit,
            resume=self.resume,
        )

        for prefix, step_key in self.step_metrics.items():
            self._run.define_metric(step_key, hidden=True)
            self._run.define_metric(f"{prefix}*", step_metric=step_key, hidden=True)

    def log(self, to_log: dict[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            raise RuntimeError("WandbLogger.log called before init()")
        del step  # Intentionally ignored; embed step in to_log instead.

        if self.warn_missing_step:
            self._check_missing_step_keys(to_log)

        self._run.log(to_log)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _check_missing_step_keys(self, to_log: dict[str, Any]) -> None:
        """Warn once per family if a to_log uses a registered prefix but omits
        its step-key. Silent fallback to the global counter is the most common
        failure mode of this design — surfacing it early saves debugging time.
        """
        for prefix, step_key in self.step_metrics.items():
            if prefix in self._warned_prefixes:
                continue
            family_keys_present = any(k.startswith(prefix) and k != step_key for k in to_log)
            if family_keys_present and step_key not in to_log:
                warnings.warn(
                    f"to_log contains keys under prefix {prefix!r} but no "
                    f"{step_key!r} entry. wandb will plot these against its "
                    f"global step counter, which may produce out-of-order "
                    f"warnings or dropped points. Add {step_key!r} to the "
                    f"to_log to use the custom step axis.",
                    stacklevel=2,
                )
                self._warned_prefixes.add(prefix)

    def watch(self, eps_model: torch.nn.Module) -> None:
        """
        watch model gradients and/or parameters.
        """
        if self.do_watch:
            if self._run is None:
                warnings.warn(
                    "Run.watch was called but 'run' is None.Execution will continue, but the model will notbe watched.",
                    stacklevel=2,
                )
                return

            self._run.watch(eps_model, log=self.watch_log, log_freq=self.watch_log_freq)
            tot_params = sum([p.numel() for p in eps_model.parameters()])
            model_cls_name = eps_model.__class__.__name__
            print(f"{model_cls_name} with {tot_params} being watched with argument log={self.watch_log} in wandb run.")

    def unwatch(self, eps_model: torch.nn.Module) -> None:

        if self.do_watch:
            if self._run is None:
                warnings.warn("Run.unwatch was called but 'run' is None.Execution will continue.", stacklevel=2)
                return

            self._run.unwatch(eps_model)
            model_cls_name = eps_model.__class__.__name__
            print(f"{model_cls_name} have been unwatched from wandb run.")
