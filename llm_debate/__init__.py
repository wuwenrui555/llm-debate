"""
llm-debate: Orchestrate multi-round debates between LLM CLI tools.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("llm-debate")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .orchestrator import DebateConfig, Orchestrator
from .participant import (
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    Participant,
    TurnContext,
    TurnResult,
)

__all__ = [
    "__version__",
    "DebateConfig",
    "Orchestrator",
    "Participant",
    "TurnContext",
    "TurnResult",
    "ClaudeParticipant",
    "CodexParticipant",
    "CustomParticipant",
]
