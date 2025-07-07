"""
Constants to store paths to vector stores
"""

from dataclasses import dataclass

LOCAL_DISK_PATH: str = './tex/RAG/local_disk'

@dataclass
class VectorStorePaths:
    INSTRUCTIONS: str = '{dir_path}/{year}_{form_name}_instruction'