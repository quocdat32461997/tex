import logging
from functools import partial
from typing import Callable, List

logger = logging.getLogger(__name__)


class BaseRegistry:
    pass


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
    def get(cls, name: str, **kwargs) -> Callable:
        assert name in cls.registry, f"Tool {name} does not exist in."  # noqa
        if name == "handoff_tool":
            return cls.registry[name](**kwargs)
        else:
            return cls.registry[name]


class ModelRegistry(BaseRegistry):
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Model %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Model {name} does not exist in."  # noqa
        return cls.registry[name]()


class ReActRegistry(BaseRegistry):
    """
    Collections of ReAct's components, including tools and models.
    Each ReAct acts according to pre-defined workflow (StateGraph).
    """

    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Model %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable:
        assert name in cls.registry, f"ReAct {name} does not exist in."  # noqa
        return partial(cls.registry[name], **kwargs)


class AgentHistory(BaseRegistry):
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Model %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @property
    def get_names_except(tbe_name_list: List[str]) -> List[str]:
        names = AgentHistory.registry.copy()
        for name in tbe_name_list:
            del names[name]

        return list(names.keys())

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable:
        assert name in cls.registry, f"ReAct {name} does not exist in."  # noqa
        return cls.registry[name](**kwargs)
