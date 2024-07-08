# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py

# pyright: strict
from collections.abc import Callable, Iterable, Iterator
from typing import Any


class Registry:
    """The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """Args:
        ----
            name (str): the name of this registry

        """
        self._name = name
        self._obj_map: dict[str, Callable[..., object] | type[str] | str] = {}

    def _do_register(
        self,
        name: str,
        obj: Callable[..., object] | type[str] | str,
        suffix: str | None = None,
    ) -> None:
        if isinstance(suffix, str):
            name = name + "_" + suffix

        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    def register(
        self,
        obj: Callable[..., object] | type[str] | None = None,
        suffix: str | None = None,
    ) -> Callable[..., object] | type[str] | str | None:
        """Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(
                func_or_class: Callable[..., object] | type[str],
            ) -> Callable[..., object] | type[str]:
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj if isinstance(obj, str) else obj.__name__
        self._do_register(name, obj, suffix)
        return None

    def get(
        self, name: str, suffix: str = "neosr"
    ) -> Callable[..., object] | type[str] | str:
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + "_" + suffix)
        if ret is None:
            msg = f"No object named '{name}' found in '{self._name}' registry!"
            raise KeyError(msg)
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self) -> Iterator[tuple[str, Callable[..., Any] | type[str] | str]]:
        return iter(self._obj_map.items())

    def keys(self) -> Iterable[str]:
        return self._obj_map.keys()


DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry("arch")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
