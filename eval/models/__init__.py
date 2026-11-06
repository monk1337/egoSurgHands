"""Hand pose model wrappers with a uniform interface."""
from .base import HandModel
from .hamer_model import HaMeRModel
from .wilor_model import WiLoRModel
from .wildhands_model import WildHandsModel

__all__ = ["HandModel", "HaMeRModel", "WiLoRModel", "WildHandsModel",
           "JointTransformerModel", "SimpleHandModel"]


def build(name: str, **kwargs) -> HandModel:
    """Factory: build a model by name."""
    name = name.lower()
    if name == "hamer":
        return HaMeRModel(**kwargs)
    if name == "wilor":
        return WiLoRModel(**kwargs)
    if name == "wildhands":
        return WildHandsModel(**kwargs)
    if name == "simplehand":
        from .simplehand_model import SimpleHandModel
        return SimpleHandModel(**kwargs)
    if name in ("jointtransformer", "jt"):
        from .jointtransformer_model import JointTransformerModel
        return JointTransformerModel(**kwargs)
    raise ValueError(f"Unknown model: {name}")
