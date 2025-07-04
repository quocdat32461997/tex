import logging
import os
from typing import Callable

import dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


class ModelFactory:
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


@ModelFactory.register("gemini_chat")
def call_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


@ModelFactory.register("gemini_embedding")
def call_gemini_embedding(
    content: str,
):
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models

    # return client.models.embed_content(
    #     model="gemini-embedding-exp-03-07",
    #     contents=content,
    # )
