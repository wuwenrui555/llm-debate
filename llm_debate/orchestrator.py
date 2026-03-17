"""
LLM debate orchestrator.

Drives multi-participant async debates with configurable rounds,
consensus detection, and structured file output. Supports resuming
from a previous session.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .participant import Participant, TurnContext

_TURN_FILE_RE = re.compile(r"(\d+)_(.+)\.md$")


@dataclass
class DebateConfig:
    """Configuration for a debate session."""

    topic: str
    participants: list[Participant]
    output_dir: Path = field(default_factory=lambda: Path("./debate_output"))
    context: str = ""
    context_file: Path | None = None
    max_rounds: int = 5
    consensus_marker: str = "[CONSENSUS REACHED]"
    turn_timeout: int = 600
    round_delay: int = 5
    resume: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.context_file and not self.context:
            self.context = Path(self.context_file).read_text(encoding="utf-8")


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


class DebateLogger:
    """Logger that writes to console and a log file."""

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def separator(self, char: str = "=", width: int = 60):
        self.log(char * width)

    def banner(self, title: str, char: str = "=", width: int = 60):
        self.log("")
        self.separator(char, width)
        padding = (width - len(title) - 2) // 2
        self.log(f"{char * padding} {title} {char * padding}")
        self.separator(char, width)


class Orchestrator:
    """
    Drives a multi-round debate between LLM participants.

    Each participant takes turns writing numbered Markdown files.
    The debate ends when consensus is detected or max rounds are reached.
    Supports resuming from existing output files.
    """

    def __init__(self, config: DebateConfig):
        self.config = config
        self.output_dir = config.output_dir

        if len(config.participants) < 2:
            raise ValueError("At least 2 participants are required")

        names = [p.name for p in config.participants]
        if len(names) != len(set(names)):
            raise ValueError("Participant names must be unique")

    def _log(self, msg: str):
        self.logger.log(msg)

    def _output_file(self, turn_index: int, participant: Participant) -> Path:
        return self.output_dir / f"{turn_index:03d}_{participant.name}.md"

    def _history_files(self) -> list[Path]:
        return sorted(
            f for f in self.output_dir.glob("*_*.md")
            if _TURN_FILE_RE.match(f.name)
        )

    def _latest_file_by(self, exclude: Participant) -> Path | None:
        """Find the latest file NOT written by the given participant."""
        files = self._history_files()
        for f in reversed(files):
            if not f.name.endswith(f"_{exclude.name}.md"):
                return f
        return None

    def _check_consensus(self) -> bool:
        files = self._history_files()
        if not files:
            return False
        content = files[-1].read_text(encoding="utf-8")
        return self.config.consensus_marker in content

    def _detect_resume_point(self) -> tuple[int, int]:
        """
        Detect where to resume from existing files.

        Returns (start_round, start_turn_index).
        Round 1 turn 1 means start from scratch.
        """
        files = self._history_files()
        if not files:
            return 1, 1

        n_participants = len(self.config.participants)

        # Parse existing turns: extract (turn_index, participant_name)
        existing_turns: list[tuple[int, str]] = []
        for f in files:
            match = _TURN_FILE_RE.match(f.name)
            if match:
                existing_turns.append((int(match.group(1)), match.group(2)))

        if not existing_turns:
            return 1, 1

        last_turn_index = existing_turns[-1][0]

        next_turn_index = last_turn_index + 1
        next_round = (next_turn_index - 1) // n_participants + 1

        return next_round, next_turn_index

    def _print_status(self, existing_files: list[Path]):
        """Print a summary of existing debate files."""
        self._log(f"Found {len(existing_files)} existing turn(s):")
        for f in existing_files:
            size = f.stat().st_size
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self._log(f"  {f.name}  ({size:,} bytes, {mtime})")

    def run(self) -> bool:
        """
        Run the debate. Returns True if consensus was reached.
        """
        cfg = self.config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = DebateLogger(self.output_dir / "debate_log.txt")

        n_participants = len(cfg.participants)
        participant_names = ", ".join(p.name for p in cfg.participants)

        # Detect resume point
        existing_files = self._history_files()
        start_round, start_turn = 1, 1

        if cfg.resume and existing_files:
            if self._check_consensus():
                self._log("Previous debate already reached consensus. Nothing to do.")
                return True
            start_round, start_turn = self._detect_resume_point()
            self.logger.banner("RESUMING DEBATE")
            self._print_status(existing_files)
            self._log(f"Resuming from round {start_round}, turn {start_turn}")
        else:
            self.logger.banner("LLM DEBATE")

        self._log(f"Topic: {cfg.topic}")
        self._log(f"Participants: {participant_names}")
        self._log(f"Max rounds: {cfg.max_rounds}")
        self._log(f"Turn timeout: {_format_duration(cfg.turn_timeout)}")
        self._log(f"Output: {self.output_dir.resolve()}")
        self._log("")

        debate_start = time.time()
        turn_index = start_turn
        total_turns_done = 0

        for round_num in range(start_round, cfg.max_rounds + 1):
            # Figure out which participant to start with in this round
            start_p_idx = (turn_index - 1) % n_participants

            self.logger.banner(f"ROUND {round_num}/{cfg.max_rounds}", char="-")

            for p_idx in range(start_p_idx, n_participants):
                participant = cfg.participants[p_idx]
                out_file = self._output_file(turn_index, participant)

                # Skip if file already exists (resume scenario)
                if out_file.exists():
                    self._log(f"  [skip] {out_file.name} already exists")
                    turn_index += 1
                    continue

                latest_opponent = self._latest_file_by(participant)

                ctx = TurnContext(
                    round_num=round_num,
                    turn_index=turn_index,
                    output_file=out_file,
                    output_dir=self.output_dir,
                    topic=cfg.topic,
                    context=cfg.context,
                    history_files=self._history_files(),
                    latest_opponent_file=latest_opponent,
                    consensus_marker=cfg.consensus_marker,
                )

                # Show who is speaking and what they're responding to
                responding_to = ""
                if latest_opponent:
                    responding_to = f" (responding to {latest_opponent.name})"

                self._log(
                    f"  [{participant.name}] Starting turn {turn_index}{responding_to}..."
                )
                self._log(f"  [{participant.name}] Output -> {out_file.name}")

                turn_start = time.time()
                result = participant.run(ctx, timeout=cfg.turn_timeout)
                turn_elapsed = time.time() - turn_start

                if not result.success:
                    self._log(
                        f"  [{participant.name}] FAILED after "
                        f"{_format_duration(turn_elapsed)}"
                    )
                    self._log(f"  [{participant.name}] Error: {result.error}")
                    if result.stderr:
                        # Show first few lines of stderr for diagnostics
                        stderr_preview = result.stderr.strip().split("\n")[:5]
                        for line in stderr_preview:
                            self._log(f"  [{participant.name}] stderr: {line}")
                    total_elapsed = time.time() - debate_start
                    self._log(
                        f"\nDebate aborted. "
                        f"Total time: {_format_duration(total_elapsed)}, "
                        f"turns completed: {total_turns_done}"
                    )
                    return False

                # Report success with file size
                file_size = out_file.stat().st_size if out_file.exists() else 0
                self._log(
                    f"  [{participant.name}] Done in "
                    f"{_format_duration(turn_elapsed)} "
                    f"({file_size:,} bytes)"
                )

                total_turns_done += 1

                if self._check_consensus():
                    total_elapsed = time.time() - debate_start
                    self.logger.banner("CONSENSUS REACHED")
                    self._log(
                        f"Consensus after {round_num} round(s), "
                        f"{total_turns_done} turn(s), "
                        f"total time: {_format_duration(total_elapsed)}"
                    )
                    self._log(f"Final output: {out_file.name}")
                    return True

                turn_index += 1

            # Round summary
            self._log(f"\n  Round {round_num} complete.")

            if round_num < cfg.max_rounds:
                self._log(
                    f"  Next round in {cfg.round_delay}s...\n"
                )
                time.sleep(cfg.round_delay)

        total_elapsed = time.time() - debate_start
        self.logger.banner("DEBATE ENDED")
        self._log(
            f"Max rounds ({cfg.max_rounds}) reached without full consensus."
        )
        self._log(
            f"Total time: {_format_duration(total_elapsed)}, "
            f"turns completed: {total_turns_done}"
        )
        self._log(f"Review the latest files in: {self.output_dir.resolve()}")
        return False
