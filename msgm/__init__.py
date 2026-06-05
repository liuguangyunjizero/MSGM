"""MSGM package."""

from importlib import import_module

__all__ = ["MSGM", "MambaConfig", "count_parameters", "make_multiscale_rpsd"]

_MODEL_EXPORTS = {"MSGM", "MambaConfig", "count_parameters"}
_PREPROCESSING_EXPORTS = {"make_multiscale_rpsd"}


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        return getattr(import_module(".model", __name__), name)
    if name in _PREPROCESSING_EXPORTS:
        return getattr(import_module(".preprocessing", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
