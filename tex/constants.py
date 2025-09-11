from dataclasses import dataclass

PATH_TO_ENV = "tex/.env"


@dataclass
class STATUS:
    SUCCESS: str = "SUCCESS"
    NEEDS_INFO: str = "NEEDS_INFO"
    IN_PROGRESS: str = "IN_PROGRESS"
    ERROR: str = "ERROR"
    HANDOFF: str = "HANDOFF"


NUM_TRIALS: int = 3
