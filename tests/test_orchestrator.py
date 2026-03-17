"""Tests for orchestrator module."""

import pytest
from pathlib import Path

from llm_debate.orchestrator import DebateConfig, Orchestrator, _format_duration
from llm_debate.participant import ClaudeParticipant, CodexParticipant, CustomParticipant


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5.123) == "5.1s"
        assert _format_duration(0.5) == "0.5s"
        assert _format_duration(59.9) == "59.9s"

    def test_minutes(self):
        assert _format_duration(60) == "1m0s"
        assert _format_duration(90) == "1m30s"
        assert _format_duration(125.7) == "2m6s"


class TestDebateConfig:
    def test_context_file(self, tmp_path):
        ctx_file = tmp_path / "context.md"
        ctx_file.write_text("My context", encoding="utf-8")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            context_file=ctx_file,
        )
        assert cfg.context == "My context"

    def test_context_string_takes_precedence(self, tmp_path):
        ctx_file = tmp_path / "context.md"
        ctx_file.write_text("File context", encoding="utf-8")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            context="Direct context",
            context_file=ctx_file,
        )
        assert cfg.context == "Direct context"


class TestOrchestratorValidation:
    def test_minimum_participants(self):
        with pytest.raises(ValueError, match="At least 2"):
            Orchestrator(DebateConfig(
                topic="test",
                participants=[ClaudeParticipant(name="claude")],
            ))

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="unique"):
            Orchestrator(DebateConfig(
                topic="test",
                participants=[
                    ClaudeParticipant(name="claude"),
                    CodexParticipant(name="claude"),
                ],
            ))


class TestResumeDetection:
    def test_empty_dir(self, tmp_path):
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
            resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._detect_resume_point() == (1, 1)

    def test_one_file_exists(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("round 1")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
            resume=True,
        )
        orch = Orchestrator(cfg)
        round_num, turn = orch._detect_resume_point()
        assert turn == 2  # Next turn after 001

    def test_full_round_exists(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("round 1")
        (tmp_path / "002_codex.md").write_text("round 1")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
            resume=True,
        )
        orch = Orchestrator(cfg)
        round_num, turn = orch._detect_resume_point()
        assert round_num == 2
        assert turn == 3

    def test_consensus_already_reached(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("[CONSENSUS REACHED]")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
            resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._check_consensus() is True

    def test_no_consensus(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("Still debating")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
            resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._check_consensus() is False


class TestFileMatching:
    def test_latest_file_excludes_self(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("a")
        (tmp_path / "002_codex.md").write_text("b")
        (tmp_path / "003_claude.md").write_text("c")
        cfg = DebateConfig(
            topic="test",
            participants=[
                ClaudeParticipant(name="claude"),
                CodexParticipant(name="codex"),
            ],
            output_dir=tmp_path,
        )
        orch = Orchestrator(cfg)
        # Claude's latest opponent file should be codex's
        result = orch._latest_file_by(cfg.participants[0])
        assert result.name == "002_codex.md"

    def test_no_false_substring_match(self, tmp_path):
        """Ensure 'bot' doesn't match 'chatbot'."""
        (tmp_path / "001_chatbot.md").write_text("a")
        (tmp_path / "002_bot.md").write_text("b")
        cfg = DebateConfig(
            topic="test",
            participants=[
                CustomParticipant(name="chatbot", cmd_template=["echo", "{prompt_file}"]),
                CustomParticipant(name="bot", cmd_template=["echo", "{prompt_file}"]),
            ],
            output_dir=tmp_path,
        )
        orch = Orchestrator(cfg)
        # bot's latest opponent should be chatbot, not excluded
        result = orch._latest_file_by(cfg.participants[1])
        assert result.name == "001_chatbot.md"

        # chatbot's latest opponent should be bot
        result = orch._latest_file_by(cfg.participants[0])
        assert result.name == "002_bot.md"
