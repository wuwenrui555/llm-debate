"""Tests for orchestrator module."""

import pytest
from pathlib import Path
from unittest.mock import patch

from llm_debate.orchestrator import DebateConfig, Orchestrator, _format_duration
from llm_debate.participant import (
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    TurnResult,
)


def _make_participants():
    return [
        ClaudeParticipant(name="claude"),
        CodexParticipant(name="codex"),
    ]


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
            participants=_make_participants(),
            context_file=ctx_file,
        )
        assert cfg.context == "My context"

    def test_context_string_takes_precedence(self, tmp_path):
        ctx_file = tmp_path / "context.md"
        ctx_file.write_text("File context", encoding="utf-8")
        cfg = DebateConfig(
            topic="test",
            participants=_make_participants(),
            context="Direct context",
            context_file=ctx_file,
        )
        assert cfg.context == "Direct context"

    def test_invalid_max_rounds(self):
        with pytest.raises(ValueError, match="max_rounds"):
            DebateConfig(topic="t", participants=_make_participants(), max_rounds=0)

    def test_invalid_turn_timeout(self):
        with pytest.raises(ValueError, match="turn_timeout"):
            DebateConfig(topic="t", participants=_make_participants(), turn_timeout=-1)

    def test_invalid_round_delay(self):
        with pytest.raises(ValueError, match="round_delay"):
            DebateConfig(topic="t", participants=_make_participants(), round_delay=-1)

    def test_round_delay_zero_is_valid(self):
        cfg = DebateConfig(topic="t", participants=_make_participants(), round_delay=0)
        assert cfg.round_delay == 0

    def test_context_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Context file not found"):
            DebateConfig(
                topic="t",
                participants=_make_participants(),
                context_file=tmp_path / "nonexistent.md",
            )


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
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._detect_resume_point() == (1, 1)

    def test_one_file_exists(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("round 1")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=True,
        )
        orch = Orchestrator(cfg)
        round_num, turn = orch._detect_resume_point()
        assert turn == 2

    def test_full_round_exists(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("round 1")
        (tmp_path / "002_codex.md").write_text("round 1")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=True,
        )
        orch = Orchestrator(cfg)
        round_num, turn = orch._detect_resume_point()
        assert round_num == 2
        assert turn == 3

    def test_consensus_already_reached(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("[CONSENSUS REACHED]")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._check_consensus() is True

    def test_no_consensus(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("Still debating")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=True,
        )
        orch = Orchestrator(cfg)
        assert orch._check_consensus() is False


class TestFileMatching:
    def test_latest_file_excludes_self(self, tmp_path):
        (tmp_path / "001_claude.md").write_text("a")
        (tmp_path / "002_codex.md").write_text("b")
        (tmp_path / "003_claude.md").write_text("c")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path,
        )
        orch = Orchestrator(cfg)
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
        result = orch._latest_file_by(cfg.participants[1])
        assert result.name == "001_chatbot.md"
        result = orch._latest_file_by(cfg.participants[0])
        assert result.name == "002_bot.md"

    def test_suffix_overlap_names(self, tmp_path):
        """Participant 'x' must not match files from 'codex'."""
        (tmp_path / "001_codex.md").write_text("a")
        (tmp_path / "002_x.md").write_text("b")
        cfg = DebateConfig(
            topic="test",
            participants=[
                CustomParticipant(name="codex", cmd_template=["echo", "{prompt_file}"]),
                CustomParticipant(name="x", cmd_template=["echo", "{prompt_file}"]),
            ],
            output_dir=tmp_path,
        )
        orch = Orchestrator(cfg)
        # x's opponent should be codex (not excluded)
        result = orch._latest_file_by(cfg.participants[1])
        assert result.name == "001_codex.md"
        # codex's opponent should be x
        result = orch._latest_file_by(cfg.participants[0])
        assert result.name == "002_x.md"

    def test_non_turn_md_files_ignored(self, tmp_path):
        """Files like notes_something.md should not be picked up."""
        (tmp_path / "001_claude.md").write_text("a")
        (tmp_path / "notes_draft.md").write_text("not a turn")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path,
        )
        orch = Orchestrator(cfg)
        files = orch._history_files()
        assert len(files) == 1
        assert files[0].name == "001_claude.md"


class TestStaleFileCleanup:
    def test_fresh_run_cleans_stale_files(self, tmp_path):
        """Non-resume run should remove stale output files."""
        (tmp_path / "001_claude.md").write_text("stale")
        (tmp_path / "002_codex.md").write_text("stale")
        cfg = DebateConfig(
            topic="test", participants=_make_participants(),
            output_dir=tmp_path, resume=False,
        )
        orch = Orchestrator(cfg)
        orch.output_dir.mkdir(parents=True, exist_ok=True)

        from llm_debate.orchestrator import DebateLogger
        import tempfile
        orch.logger = DebateLogger(Path(tempfile.mktemp()))

        orch._clean_stale_files()
        assert orch._history_files() == []


class TestOrchestratorRun:
    """Integration tests for the full run loop using mocked participants."""

    def _make_mock_participant(self, name, write_content="output"):
        """Create a CustomParticipant with a mocked run method."""
        p = CustomParticipant(name=name, cmd_template=["echo", "{prompt_file}"])

        def mock_run(ctx, *, timeout=600):
            ctx.output_file.write_text(write_content, encoding="utf-8")
            return TurnResult(success=True)

        p.run = mock_run
        return p

    def test_consensus_in_round_1(self, tmp_path):
        p1 = self._make_mock_participant("alice", "Hello")
        p2 = self._make_mock_participant("bob", "[CONSENSUS REACHED] We agree!")

        cfg = DebateConfig(
            topic="test",
            participants=[p1, p2],
            output_dir=tmp_path,
            max_rounds=3,
            round_delay=0,
        )
        orch = Orchestrator(cfg)
        result = orch.run()
        assert result is True
        assert (tmp_path / "001_alice.md").exists()
        assert (tmp_path / "002_bob.md").exists()

    def test_max_rounds_no_consensus(self, tmp_path):
        p1 = self._make_mock_participant("alice", "I disagree")
        p2 = self._make_mock_participant("bob", "I also disagree")

        cfg = DebateConfig(
            topic="test",
            participants=[p1, p2],
            output_dir=tmp_path,
            max_rounds=2,
            round_delay=0,
        )
        orch = Orchestrator(cfg)
        result = orch.run()
        assert result is False
        # 2 rounds * 2 participants = 4 files
        assert len(orch._history_files()) == 4

    def test_participant_failure_aborts(self, tmp_path):
        p1 = self._make_mock_participant("alice", "Hello")
        p2 = CustomParticipant(name="bob", cmd_template=["echo", "{prompt_file}"])

        def fail_run(ctx, *, timeout=600):
            return TurnResult(success=False, error="CLI crashed")

        p2.run = fail_run

        cfg = DebateConfig(
            topic="test",
            participants=[p1, p2],
            output_dir=tmp_path,
            max_rounds=3,
            round_delay=0,
        )
        orch = Orchestrator(cfg)
        result = orch.run()
        assert result is False
        assert (tmp_path / "001_alice.md").exists()
        assert not (tmp_path / "002_bob.md").exists()

    def test_resume_skips_existing(self, tmp_path):
        # Pre-create first turn
        (tmp_path / "001_alice.md").write_text("pre-existing")

        p1 = self._make_mock_participant("alice", "new output")
        p2 = self._make_mock_participant("bob", "[CONSENSUS REACHED]")

        cfg = DebateConfig(
            topic="test",
            participants=[p1, p2],
            output_dir=tmp_path,
            max_rounds=3,
            round_delay=0,
            resume=True,
        )
        orch = Orchestrator(cfg)
        result = orch.run()
        assert result is True
        # Original file should be unchanged
        assert (tmp_path / "001_alice.md").read_text() == "pre-existing"
        assert (tmp_path / "002_bob.md").exists()
