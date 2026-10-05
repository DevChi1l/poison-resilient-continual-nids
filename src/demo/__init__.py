"""Local, artifact-backed demonstration helpers.

The package intentionally keeps model imports lazy so saved-result playback,
schema checks, and quarantine review remain usable without PyTorch installed.
"""

from .artifacts import (
    ArtifactConfigurationError,
    DemoArtifactConfig,
    ModelArtifact,
    load_demo_config,
    load_fitted_preprocessor,
)

__all__ = [
    "ArtifactConfigurationError",
    "DemoArtifactConfig",
    "ModelArtifact",
    "load_demo_config",
    "load_fitted_preprocessor",
]
