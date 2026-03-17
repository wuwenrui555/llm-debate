"""
llm-debate: Orchestrate multi-round debates between LLM CLI tools.
"""

from .orchestrator import DebateConfig, Orchestrator
from .participant import (
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    Participant,
    TurnContext,
)

__all__ = [
    "DebateConfig",
    "Orchestrator",
    "Participant",
    "TurnContext",
    "ClaudeParticipant",
    "CodexParticipant",
    "CustomParticipant",
]
