import os

import dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()

from tex.models.model_registry import ModelRegistry


@ModelRegistry.register("gemini_chat")
def call_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


@ModelRegistry.register("gemini_embedding")
def call_gemini_embedding(
    content: str,
):
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models

    # return client.models.embed_content(
    #     model="gemini-embedding-exp-03-07",
    #     contents=content,
    # )
