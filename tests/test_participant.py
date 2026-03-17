"""Tests for participant module."""

import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm_debate.participant import (
    Participant,
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    TurnContext,
    TurnResult,
)


def _make_ctx(tmp_path, **kwargs) -> TurnContext:
    defaults = dict(
        round_num=1,
        turn_index=1,
        output_file=tmp_path / "001_test.md",
        output_dir=tmp_path,
        topic="Test topic",
        context="",
        history_files=[],
        latest_opponent_file=None,
        consensus_marker="[CONSENSUS REACHED]",
    )
    defaults.update(kwargs)
    return TurnContext(**defaults)


class TestParticipantNameValidation:
    def test_valid_names(self):
        ClaudeParticipant(name="claude")
        ClaudeParticipant(name="claude-v2")
        ClaudeParticipant(name="my_bot_3")

    def test_invalid_name_slash(self):
        with pytest.raises(ValueError, match="Invalid participant name"):
            ClaudeParticipant(name="../../etc")

    def test_invalid_name_dot(self):
        with pytest.raises(ValueError, match="Invalid participant name"):
            ClaudeParticipant(name="..bad")

    def test_invalid_name_space(self):
        with pytest.raises(ValueError, match="Invalid participant name"):
            ClaudeParticipant(name="my bot")

    def test_invalid_name_empty(self):
        with pytest.raises(ValueError, match="Invalid participant name"):
            ClaudeParticipant(name="")


class TestBuildPrompt:
    def test_first_round_prompt(self, tmp_path):
        p = ClaudeParticipant(name="claude")
        ctx = _make_ctx(tmp_path)
        prompt = p.build_prompt(ctx)
        assert "first round" in prompt
        assert "AWAITING REVIEW" in prompt
        assert "Test topic" in prompt

    def test_subsequent_round_prompt(self, tmp_path):
        p = ClaudeParticipant(name="claude")
        ctx = _make_ctx(
            tmp_path,
            latest_opponent_file=Path("/tmp/01_codex.md"),
        )
        prompt = p.build_prompt(ctx)
        assert "Previous review" in prompt
        assert "01_codex.md" in prompt
        assert "[CONSENSUS REACHED]" in prompt

    def test_context_included(self, tmp_path):
        p = ClaudeParticipant(name="claude")
        ctx = _make_ctx(tmp_path, context="Important background info")
        prompt = p.build_prompt(ctx)
        assert "Important background info" in prompt


class TestBuildCommand:
    def test_claude_command(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt", encoding="utf-8")
        p = ClaudeParticipant(name="claude", cmd="/usr/bin/claude")
        cmd = p.build_command(prompt_file, Path("/work"))
        assert cmd[0] == "/usr/bin/claude"
        assert "--dangerously-skip-permissions" in cmd
        assert "-p" in cmd
        assert "test prompt" in cmd

    def test_codex_command(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("test prompt", encoding="utf-8")
        p = CodexParticipant(name="codex", cmd="/usr/bin/codex")
        cmd = p.build_command(prompt_file, Path("/work"))
        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "/work" in cmd
        assert "test prompt" in cmd

    def test_custom_command_prompt_file(self):
        p = CustomParticipant(
            name="mybot",
            cmd_template=["mybot", "--file", "{prompt_file}"],
        )
        cmd = p.build_command(Path("/tmp/prompt.txt"), Path("/work"))
        assert cmd == ["mybot", "--file", "/tmp/prompt.txt"]

    def test_custom_command_prompt_backward_compat(self):
        p = CustomParticipant(
            name="mybot",
            cmd_template=["mybot", "--run", "{prompt}"],
        )
        cmd = p.build_command(Path("/tmp/prompt.txt"), Path("/work"))
        assert cmd == ["mybot", "--run", "/tmp/prompt.txt"]


class TestParticipantRun:
    """Tests for Participant.run() subprocess execution."""

    def test_successful_run(self, tmp_path):
        """Subprocess succeeds and creates output file."""
        out_file = tmp_path / "001_mybot.md"
        ctx = _make_ctx(tmp_path, output_file=out_file)

        # Use a real command that writes the output file
        p = CustomParticipant(
            name="mybot",
            cmd_template=["bash", "-c", f"echo 'output' > {out_file}"],
        )
        # Override build_command to ignore prompt_file
        original_build = p.build_command

        def fixed_build(prompt_file, cwd):
            return ["bash", "-c", f"echo 'output' > {out_file}"]

        p.build_command = fixed_build
        result = p.run(ctx, timeout=10)
        assert result.success is True
        assert out_file.exists()

    def test_timeout_kills_process(self, tmp_path):
        """Subprocess that times out returns failure."""
        out_file = tmp_path / "001_mybot.md"
        ctx = _make_ctx(tmp_path, output_file=out_file)

        p = CustomParticipant(
            name="mybot",
            cmd_template=["sleep", "999"],
        )
        # Override to not use {prompt_file} in the command
        p.build_command = lambda pf, cwd: ["sleep", "999"]

        result = p.run(ctx, timeout=1)
        assert result.success is False
        assert "Timed out" in result.error

    def test_process_failure_captures_stderr(self, tmp_path):
        """Failed subprocess captures stderr."""
        out_file = tmp_path / "001_mybot.md"
        ctx = _make_ctx(tmp_path, output_file=out_file)

        p = CustomParticipant(
            name="mybot",
            cmd_template=["bash", "-c", "echo 'error msg' >&2; exit 1"],
        )
        p.build_command = lambda pf, cwd: ["bash", "-c", "echo 'error msg' >&2; exit 1"]

        result = p.run(ctx, timeout=10)
        assert result.success is False
        assert result.return_code == 1
        assert "error msg" in result.stderr

    def test_prompt_file_cleaned_up(self, tmp_path):
        """Prompt temp file is removed after run."""
        out_file = tmp_path / "001_mybot.md"
        ctx = _make_ctx(tmp_path, output_file=out_file)

        p = CustomParticipant(
            name="mybot",
            cmd_template=["bash", "-c", f"echo 'ok' > {out_file}"],
        )
        p.build_command = lambda pf, cwd: ["bash", "-c", f"echo 'ok' > {out_file}"]

        result = p.run(ctx, timeout=10)
        assert result.success is True
        # No temp files should remain
        import glob
        temps = glob.glob("/tmp/llm_debate_mybot_*")
        assert len(temps) == 0

    def test_missing_output_file(self, tmp_path):
        """Process succeeds but doesn't create the expected file."""
        out_file = tmp_path / "001_mybot.md"
        ctx = _make_ctx(tmp_path, output_file=out_file)

        p = CustomParticipant(
            name="mybot",
            cmd_template=["true"],
        )
        p.build_command = lambda pf, cwd: ["true"]

        # Patch time.sleep to avoid 30s wait
        with patch("llm_debate.participant.time.sleep"):
            result = p.run(ctx, timeout=10)
        assert result.success is False
        assert "not created" in result.error
