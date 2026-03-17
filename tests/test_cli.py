"""Tests for CLI module."""

import pytest

from llm_debate.cli import parse_custom_cmds


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

    def test_odd_count_exits(self):
        with pytest.raises(SystemExit):
            parse_custom_cmds(["mybot"])
