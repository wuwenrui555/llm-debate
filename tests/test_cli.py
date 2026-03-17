"""Tests for CLI module."""

import pytest

from llm_debate.cli import parse_custom_cmds, build_participants, BUILTIN_PARTICIPANTS


class TestParseCustomCmds:
    def test_empty(self):
        assert parse_custom_cmds(None) == {}
        assert parse_custom_cmds([]) == {}

    def test_single_pair(self):
        result = parse_custom_cmds(["mybot", "mybot-cli --auto {prompt_file}"])
        assert result == {"mybot": ["mybot-cli", "--auto", "{prompt_file}"]}

    def test_quoted_args_handled(self):
        result = parse_custom_cmds(["mybot", "mybot-cli --arg 'hello world' {prompt_file}"])
        assert result == {"mybot": ["mybot-cli", "--arg", "hello world", "{prompt_file}"]}

    def test_odd_count_raises(self):
        with pytest.raises(ValueError, match="pairs"):
            parse_custom_cmds(["mybot"])

    def test_multiple_pairs(self):
        result = parse_custom_cmds([
            "bot1", "cli1 {prompt_file}",
            "bot2", "cli2 --flag {prompt_file}",
        ])
        assert "bot1" in result
        assert "bot2" in result


class TestBuildParticipants:
    def test_builtin_claude(self):
        participants = build_participants(["claude"], {})
        assert len(participants) == 1
        assert participants[0].name == "claude"

    def test_builtin_codex(self):
        participants = build_participants(["codex"], {})
        assert len(participants) == 1
        assert participants[0].name == "codex"

    def test_custom_participant(self):
        custom = {"mybot": ["mybot-cli", "{prompt_file}"]}
        participants = build_participants(["mybot"], custom)
        assert len(participants) == 1
        assert participants[0].name == "mybot"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown participant"):
            build_participants(["nonexistent"], {})

    def test_mixed_builtin_and_custom(self):
        custom = {"mybot": ["mybot-cli", "{prompt_file}"]}
        participants = build_participants(["claude", "mybot"], custom)
        assert len(participants) == 2
        assert participants[0].name == "claude"
        assert participants[1].name == "mybot"
