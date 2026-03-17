"""Tests for participant module."""

import pytest
from pathlib import Path

from llm_debate.participant import (
    Participant,
    ClaudeParticipant,
    CodexParticipant,
    CustomParticipant,
    TurnContext,
)


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
    def _make_ctx(self, **kwargs) -> TurnContext:
        defaults = dict(
            round_num=1,
            turn_index=1,
            output_file=Path("/tmp/01_test.md"),
            output_dir=Path("/tmp"),
            topic="Test topic",
            context="",
            history_files=[],
            latest_opponent_file=None,
            consensus_marker="[CONSENSUS REACHED]",
        )
        defaults.update(kwargs)
        return TurnContext(**defaults)

    def test_first_round_prompt(self):
        p = ClaudeParticipant(name="claude")
        ctx = self._make_ctx()
        prompt = p.build_prompt(ctx)
        assert "first round" in prompt
        assert "AWAITING REVIEW" in prompt
        assert "Test topic" in prompt

    def test_subsequent_round_prompt(self):
        p = ClaudeParticipant(name="claude")
        ctx = self._make_ctx(
            latest_opponent_file=Path("/tmp/01_codex.md"),
        )
        prompt = p.build_prompt(ctx)
        assert "Previous review" in prompt
        assert "01_codex.md" in prompt
        assert "[CONSENSUS REACHED]" in prompt

    def test_context_included(self):
        p = ClaudeParticipant(name="claude")
        ctx = self._make_ctx(context="Important background info")
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
