import logging
from typing import Callable

from tex.registry import BaseRegistry

logger = logging.getLogger(__name__)


class ToolRegistry(BaseRegistry):
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Tool %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Tool {name} does not exist in."  # noqa
        return cls.registry[name]
