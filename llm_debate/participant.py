"""
LLM debate participants.

Each participant wraps a CLI command that accepts a prompt
and writes its response to a designated output file.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TurnContext:
    """Information passed to a participant for each turn."""

    round_num: int
    turn_index: int
    output_file: Path
    output_dir: Path
    topic: str
    context: str
    history_files: list[Path]
    latest_opponent_file: Path | None
    consensus_marker: str


class Participant(ABC):
    """Base class for a debate participant."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        """Return the CLI command list to execute."""

    def build_prompt(self, ctx: TurnContext) -> str:
        """Build the prompt for this turn. Override for custom behavior."""
        parts: list[str] = [
            f"You are a research consultant AI ({self.name}), "
            f"participating in an async review debate.",
            "",
            f"## Topic",
            ctx.topic,
            "",
        ]

        if ctx.context:
            parts += [f"## Context", ctx.context, ""]

        if ctx.latest_opponent_file:
            parts += [
                f"## Previous review to respond to",
                f"Please read file: {ctx.latest_opponent_file}",
                "",
                f"## Your task",
                f"1. Respond to each point (agree / disagree / supplement)",
                f"2. Propose your own approach",
                f"3. List consensus points and disagreements",
                f"4. If core consensus is reached, write {ctx.consensus_marker} "
                f"and output a consensus summary",
                f"5. Otherwise write [AWAITING REVIEW] and list open questions",
            ]
        else:
            parts += [
                f"## Your task (first round)",
                f"1. Read all relevant files in {ctx.output_dir}",
                f"2. Analyze and propose your initial approach",
                f"3. List your priorities",
                f"4. End with [AWAITING REVIEW]",
            ]

        parts += [
            "",
            f"## Output requirements",
            f"Markdown format, write to file: {ctx.output_file}",
        ]

        return "\n".join(parts)

    def run(self, ctx: TurnContext, *, timeout: int = 600) -> bool:
        """Execute the participant and wait for output file."""
        prompt = self.build_prompt(ctx)
        cmd = self.build_command(prompt, ctx.output_dir)

        try:
            subprocess.run(
                cmd,
                cwd=ctx.output_dir,
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        except subprocess.CalledProcessError:
            return False

        # Wait for file to appear (CLI may flush after exit)
        for _ in range(15):
            if ctx.output_file.exists():
                return True
            time.sleep(2)

        return False


class ClaudeParticipant(Participant):
    """Participant backed by Claude Code CLI."""

    def __init__(self, name: str = "claude", cmd: str | None = None):
        super().__init__(name)
        self.cmd = cmd or shutil.which("claude") or shutil.which("claude-code") or "claude"

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        return [self.cmd, "--dangerously-skip-permissions", "-p", prompt]


class CodexParticipant(Participant):
    """Participant backed by OpenAI Codex CLI."""

    def __init__(self, name: str = "codex", cmd: str | None = None):
        super().__init__(name)
        self.cmd = cmd or shutil.which("codex") or shutil.which("openai-codex") or "codex"

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        return [
            self.cmd, "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C", str(cwd),
            prompt,
        ]


class CustomParticipant(Participant):
    """
    Participant backed by an arbitrary shell command template.

    The template should contain ``{prompt}`` as a placeholder.
    Example: ``["my-llm", "--run", "{prompt}"]``
    """

    def __init__(self, name: str, cmd_template: list[str]):
        super().__init__(name)
        self.cmd_template = cmd_template

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        return [part.replace("{prompt}", prompt) for part in self.cmd_template]
