"""
CLI entry point for llm-debate.

Usage:
    llm-debate --participants claude codex \
               --topic "Review my API design" \
               --max-rounds 5 \
               --output-dir ./debate_output

    llm-debate --participants claude codex gemini \
               --custom-cmd gemini "gemini-cli --auto {prompt_file}" \
               --topic "Architecture review" \
               --context-file ./context.md
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from .orchestrator import DebateConfig, Orchestrator
from .participant import (
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    Participant,
)

BUILTIN_PARTICIPANTS: dict[str, type[Participant]] = {
    "claude": ClaudeParticipant,
    "codex": CodexParticipant,
}


def parse_custom_cmds(raw: list[str] | None) -> dict[str, list[str]]:
    """
    Parse --custom-cmd pairs: name "cmd --flag {prompt_file}"
    Returns {name: [cmd_template_parts]}.
    """
    if not raw:
        return {}
    if len(raw) % 2 != 0:
        print("Error: --custom-cmd requires pairs of <name> <command>", file=sys.stderr)
        sys.exit(1)
    result = {}
    for i in range(0, len(raw), 2):
        name = raw[i]
        cmd_str = raw[i + 1]
        result[name] = shlex.split(cmd_str)
    return result


def build_participants(
    names: list[str],
    custom_cmds: dict[str, list[str]],
) -> list[Participant]:
    """Instantiate participants by name, using built-ins or custom commands."""
    participants = []
    for name in names:
        if name in BUILTIN_PARTICIPANTS:
            participants.append(BUILTIN_PARTICIPANTS[name](name=name))
        elif name in custom_cmds:
            participants.append(CustomParticipant(name=name, cmd_template=custom_cmds[name]))
        else:
            print(
                f"Error: unknown participant '{name}'. "
                f"Use a built-in ({', '.join(BUILTIN_PARTICIPANTS)}) "
                f"or define it with --custom-cmd {name} '<command>'",
                file=sys.stderr,
            )
            sys.exit(1)
    return participants


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="llm-debate",
        description="Orchestrate multi-round debates between LLM CLI tools",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        required=True,
        help="Participant names (built-in: claude, codex; or custom via --custom-cmd)",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="The debate topic / question",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Additional context string for the debate",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Path to a file containing additional context",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./debate_output"),
        help="Directory for output files (default: ./debate_output)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of debate rounds (default: 5)",
    )
    parser.add_argument(
        "--consensus-marker",
        default="[CONSENSUS REACHED]",
        help="String that signals consensus (default: [CONSENSUS REACHED])",
    )
    parser.add_argument(
        "--turn-timeout",
        type=int,
        default=600,
        help="Timeout per turn in seconds (default: 600)",
    )
    parser.add_argument(
        "--round-delay",
        type=int,
        default=5,
        help="Delay between rounds in seconds (default: 5)",
    )
    parser.add_argument(
        "--custom-cmd",
        nargs="*",
        metavar="NAME CMD",
        help='Define custom participants: --custom-cmd mybot "mybot-cli --auto {prompt_file}"',
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing output files instead of starting fresh",
    )

    args = parser.parse_args(argv)

    custom_cmds = parse_custom_cmds(args.custom_cmd)
    participants = build_participants(args.participants, custom_cmds)

    config = DebateConfig(
        topic=args.topic,
        participants=participants,
        output_dir=args.output_dir,
        context=args.context,
        context_file=args.context_file,
        max_rounds=args.max_rounds,
        consensus_marker=args.consensus_marker,
        turn_timeout=args.turn_timeout,
        round_delay=args.round_delay,
        resume=args.resume,
    )

    orchestrator = Orchestrator(config)
    consensus = orchestrator.run()
    sys.exit(0 if consensus else 1)


if __name__ == "__main__":
    main()
