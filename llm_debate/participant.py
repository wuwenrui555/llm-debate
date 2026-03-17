"""
LLM debate participants.

Each participant wraps a CLI command that accepts a prompt
and writes its response to a designated output file.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


@dataclass
class TurnResult:
    """Result of a participant's turn."""

    success: bool
    error: str | None = None
    return_code: int | None = None
    stderr: str | None = None


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
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid participant name '{name}': "
                f"must be alphanumeric, hyphens, or underscores only"
            )
        self.name = name

    @abstractmethod
    def build_command(self, prompt_file: Path, cwd: Path) -> list[str]:
        """Return the CLI command list to execute.

        The prompt is written to a temporary file; use prompt_file
        to reference it in the command.
        """

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

    def run(self, ctx: TurnContext, *, timeout: int = 600) -> TurnResult:
        """Execute the participant and wait for output file."""
        prompt = self.build_prompt(ctx)

        # Write prompt to a temp file to avoid exposing it in process args
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix=f"llm_debate_{self.name}_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(prompt)
            prompt_file = Path(f.name)

        try:
            cmd = self.build_command(prompt_file, ctx.output_dir)
            result = subprocess.run(
                cmd,
                cwd=ctx.output_dir,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return TurnResult(
                success=False,
                error=f"Timed out after {timeout}s",
            )
        except subprocess.CalledProcessError as e:
            return TurnResult(
                success=False,
                error=f"Process exited with code {e.returncode}",
                return_code=e.returncode,
                stderr=(e.stderr or "")[:2000],
            )
        finally:
            prompt_file.unlink(missing_ok=True)

        # Wait for file to appear (CLI may flush after exit)
        for _ in range(15):
            if ctx.output_file.exists():
                return TurnResult(success=True)
            time.sleep(2)

        return TurnResult(
            success=False,
            error=f"Output file {ctx.output_file.name} not created after process exited",
        )


class ClaudeParticipant(Participant):
    """Participant backed by Claude Code CLI.

    WARNING: Uses --dangerously-skip-permissions which bypasses all
    permission checks. Only use in trusted/sandboxed environments.
    """

    def __init__(self, name: str = "claude", cmd: str | None = None):
        super().__init__(name)
        self.cmd = cmd or shutil.which("claude") or shutil.which("claude-code") or "claude"

    def build_command(self, prompt_file: Path, cwd: Path) -> list[str]:
        prompt = prompt_file.read_text(encoding="utf-8")
        return [self.cmd, "--dangerously-skip-permissions", "-p", prompt]


class CodexParticipant(Participant):
    """Participant backed by OpenAI Codex CLI.

    WARNING: Uses --dangerously-bypass-approvals-and-sandbox which
    disables all safety checks. Only use in trusted/sandboxed environments.
    """

    def __init__(self, name: str = "codex", cmd: str | None = None):
        super().__init__(name)
        self.cmd = cmd or shutil.which("codex") or shutil.which("openai-codex") or "codex"

    def build_command(self, prompt_file: Path, cwd: Path) -> list[str]:
        prompt = prompt_file.read_text(encoding="utf-8")
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

    The template should contain ``{prompt_file}`` as a placeholder for
    the path to the prompt file. The prompt content is written to a
    temporary file and the path is substituted into the command.

    Example: ``["my-llm", "--prompt-file", "{prompt_file}"]``

    WARNING: Do NOT use templates that pass {prompt_file} to a shell
    interpreter (e.g., ["sh", "-c", "cat {prompt_file} | ..."]).
    """

    def __init__(self, name: str, cmd_template: list[str]):
        super().__init__(name)
        self.cmd_template = cmd_template

    def build_command(self, prompt_file: Path, cwd: Path) -> list[str]:
        pf = str(prompt_file)
        return [
            part.replace("{prompt_file}", pf).replace("{prompt}", pf)
            for part in self.cmd_template
        ]
