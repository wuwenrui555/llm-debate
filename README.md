# llm-debate

Orchestrate multi-round debates between LLM CLI tools (Claude Code, Codex, or any custom CLI).

## Features

- **Multi-participant**: Not limited to 2 — any number of LLM participants can debate
- **Resumable**: Stop and continue debates with `--resume`
- **Extensible**: Built-in support for Claude Code and Codex, plus `CustomParticipant` for any CLI
- **Consensus detection**: Automatically stops when participants reach agreement
- **Detailed logging**: Per-turn timing, file sizes, round summaries

## Install

```bash
pip install llm-debate
```

### Prerequisites

At least two LLM CLI tools installed:

- [Claude Code](https://github.com/anthropics/claude-code): `npm install -g @anthropic-ai/claude-code`
- [Codex](https://github.com/openai/codex): `npm install -g @openai/codex`

## Quick Start

### CLI

```bash
# Two-participant debate
llm-debate --participants claude codex \
           --topic "Review this system architecture" \
           --context "We're building a distributed cache with write-through semantics..." \
           --output-dir ./debate_output \
           --max-rounds 5

# Resume after interruption
llm-debate --participants claude codex \
           --topic "Review this system architecture" \
           --output-dir ./debate_output \
           --resume

# Custom participant
llm-debate --participants claude mybot \
           --custom-cmd mybot "my-llm-cli --auto {prompt}" \
           --topic "Code review"
```

### Python API

```python
from llm_debate import Orchestrator, DebateConfig, ClaudeParticipant, CodexParticipant

debate = Orchestrator(DebateConfig(
    topic="Review my API design",
    participants=[
        ClaudeParticipant(name="claude"),
        CodexParticipant(name="codex"),
    ],
    context="We're designing a REST API for...",
    max_rounds=5,
    output_dir="./debate_output",
))
consensus = debate.run()
```

### Custom Participants

```python
from llm_debate import Orchestrator, DebateConfig, CustomParticipant, ClaudeParticipant

debate = Orchestrator(DebateConfig(
    topic="Architecture review",
    participants=[
        ClaudeParticipant(name="claude"),
        CustomParticipant(name="gemini", cmd_template=["gemini-cli", "--auto", "{prompt}"]),
    ],
))
debate.run()
```

## How It Works

1. Participants take turns writing numbered Markdown files (`01_claude.md`, `02_codex.md`, ...)
2. Each participant reads the previous opponent's file and responds
3. The debate continues until:
   - A participant writes `[CONSENSUS REACHED]` in their output
   - The maximum number of rounds is reached

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--participants` | (required) | Participant names |
| `--topic` | (required) | Debate topic |
| `--context` | `""` | Additional context |
| `--context-file` | | Load context from file |
| `--output-dir` | `./debate_output` | Output directory |
| `--max-rounds` | `5` | Max debate rounds |
| `--consensus-marker` | `[CONSENSUS REACHED]` | Consensus signal string |
| `--turn-timeout` | `600` | Per-turn timeout (seconds) |
| `--round-delay` | `5` | Delay between rounds (seconds) |
| `--resume` | `false` | Resume from existing files |
| `--custom-cmd` | | Define custom participants |

## License

MIT
