"""Stdlib-only evaluation timeout recipe; no simulator or learner imports."""
from dataclasses import asdict, dataclass
from .env_config import EnvConfig


@dataclass(frozen=True)
class EvaluationEnvConfig(EnvConfig):
    """Longer evaluation timeout only; validate every inherited physical field."""
    episode_seconds: float = 90.

    def __post_init__(self):
        if self.episode_seconds != 90.:
            raise ValueError("Evaluation timeout is explicitly fixed at 90 seconds")
        EnvConfig(**{**asdict(self), "episode_seconds": 60.})
