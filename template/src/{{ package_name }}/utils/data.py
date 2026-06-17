from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


def instantiate_dataset(cfg: DictConfig, train: bool | None) -> Dataset:
    """
    Instantiate a `Dataset` Object.
    Args:
        - cfg: Dataset configuration object of type `DictConfig`
        - train: Boolean to instantiate train or test set.
        Can be None in case no train argument have to be passed.
    """

    if cfg.get("instantiate") is None:
        raise KeyError("instantiate_dataset expects `instantiate` key in DictConfig.")

    transforms = [ToTensor()]
    if cfg.get("transforms"):
        transforms_cfg = cfg.get("transforms")
        if not isinstance(transforms_cfg, DictConfig):
            raise ValueError(
                f"Expected transforms node to be of type `Dictconfig`but got type `{type(transforms_cfg)}` instead."
            )
        transforms_dict = OmegaConf.to_container(transforms_cfg, resolve=True)
        for _transform_name, to_instantiate_obj in transforms_dict.items():
            transforms += [instantiate(to_instantiate_obj)]

    override_kwargs = {"transform": Compose(transforms), "_convert_": "full"}
    if train is None:
        return instantiate(cfg.instantiate, **override_kwargs)

    override_kwargs["train"] = train
    return instantiate(cfg.instantiate, **override_kwargs)
