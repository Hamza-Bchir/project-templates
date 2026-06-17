import warnings

import torch
from omegaconf import OmegaConf

from stadif import utils


def register_resolvers():
    if not OmegaConf.has_resolver("generate_random_seed"):
        OmegaConf.register_new_resolver("generate_random_seed", utils.seeding.generate_random_seed, use_cache=True)
    if not OmegaConf.has_resolver("resolve_device"):
        OmegaConf.register_new_resolver("resolve_device", resolve_device, use_cache=True)
    if not OmegaConf.has_resolver("resolve_model_on_device"):
        OmegaConf.register_new_resolver(
            "resolve_model_on_device",
            resolve_model_on_device,
            use_cache=True,
        )
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval, use_cache=True)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "(Ignore if overridden) Device set to CUDA, but CUDA is not available. Defaulting to CPU.", stacklevel=2
        )
        return torch.device("cpu")
    elif device == "mps" and not torch.backends.mps.is_available():
        warnings.warn(
            "(Ignore if overridden) Device set to MPS, but MPS is not available. Defaulting to CPU.", stacklevel=2
        )
        return torch.device("cpu")
    else:
        return device