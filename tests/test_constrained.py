"""Constrained decoding.

Measured on the native tool-call channel, this model produces a tool call for
80% of prompts at a four tool surface and 33% at eighteen. The rest of the time
it writes the call into its prose, and a turn with no tool calls looks exactly
like a turn that finished. Constrained against a schema the same six sizes
measure 100%.
"""

from __future__ import annotations

import json

from mimir.agent.constrained import ANSWER, build_schema, parse_step
from mimir.tools.base import load_all_tools


def _specs(*names):
    registry = load_all_tools()
    return [s for s in (registry.get(n) for n in names) if s is not None]


class TestSchema:
    def test_every_tool_becomes_a_branch_keyed_by_its_name(self):
        schema = build_schema(_specs("get_logs", "list_workloads"))
        names = [b["properties"]["tool"]["const"] for b in schema["anyOf"]]
        assert "get_logs" in names and "list_workloads" in names

    def test_each_branch_carries_that_tool_s_own_arguments(self):
        """A generic object for arguments let the model invent field names:
        pod_name and lines instead of target and tail."""
        schema = build_schema(_specs("get_logs"))
        branch = next(b for b in schema["anyOf"] if b["properties"]["tool"]["const"] == "get_logs")
        assert "target" in branch["properties"]["arguments"]["properties"]

    def test_there_is_a_way_to_stop(self):
        """A loop whose only legal move is another tool call cannot finish."""
        schema = build_schema(_specs("get_logs"))
        assert ANSWER in [b["properties"]["tool"]["const"] for b in schema["anyOf"]]

    def test_hidden_arguments_are_hidden_here_too(self):
        """Or the two decoders offer different choices and a comparison between
        them measures the surface rather than the decoder."""
        schema = build_schema(_specs("find_workloads"), hidden=("limit",))
        branch = schema["anyOf"][0]
        assert "limit" not in branch["properties"]["arguments"]["properties"]

    def test_it_is_valid_json(self):
        assert json.loads(json.dumps(build_schema(_specs("get_logs"))))


class TestParsing:
    def test_a_tool_step_becomes_a_tool_call(self):
        step = parse_step('{"say": "checking", "tool": "get_logs", '
                          '"arguments": {"target": "api"}}')
        assert not step.finished
        call = step.as_tool_call()
        assert call.name == "get_logs"
        assert call.arguments == {"target": "api"}

    def test_the_answer_branch_ends_the_turn(self):
        step = parse_step('{"say": "there are no such pods", "tool": "answer"}')
        assert step.finished
        assert step.as_tool_call() is None

    def test_an_empty_response_ends_the_turn_rather_than_raising(self):
        """A turn that produced nothing is finished whatever the reason."""
        assert parse_step("").finished

    def test_unconstrained_text_is_kept_rather_than_lost(self):
        """The constraint should make this impossible. If a runtime returns it
        anyway, losing the content would be worse than showing it."""
        step = parse_step("I could not do that.")
        assert step.finished
        assert "could not" in step.say

    def test_what_the_model_says_reaches_the_operator(self):
        step = parse_step('{"say": "looking in payments", "tool": "answer"}')
        assert step.say == "looking in payments"


class TestFinishing:
    def test_the_answer_branch_says_it_is_how_you_finish(self):
        """Under a constrained decoder the model cannot wander into prose to
        signal it is done, so the only way it learns to stop is the schema.
        Without this it retrieved what was asked for and then repeated the same
        successful call six times until the step budget ran out."""
        schema = build_schema(_specs("get_logs"))
        answer = next(
            b for b in schema["anyOf"] if b["properties"]["tool"]["const"] == ANSWER
        )
        assert "Finish" in answer["description"]
        assert "answer" in answer["properties"]["say"]["description"].lower()

    def test_a_tool_branch_says_it_continues(self):
        schema = build_schema(_specs("get_logs"))
        branch = next(
            b for b in schema["anyOf"] if b["properties"]["tool"]["const"] == "get_logs"
        )
        assert "Call get_logs" in branch["description"]


class TestTruncationIsNotAnAnswer:
    """Constrained decoding regressed the coding loop from 3/3 to 0/4, and the
    cause was a token budget. A coding tool's arguments are whole blocks of
    code; a 900 token cap cut them off mid-string, the JSON stopped parsing,
    and the truncated fragment was treated as the model's final word. The turn
    ended having changed nothing. Operations calls carry short strings and
    never hit it, which is why only one loop was affected.
    """

    def test_a_truncated_call_does_not_parse_as_a_tool_call(self):
        cut = '{"say": "editing", "tool": "edit_worktree_file", "arguments": {"old_'
        step = parse_step(cut)
        assert step.as_tool_call() is None

    def test_the_budget_comes_from_the_profile_not_a_constant(self):
        import inspect as _inspect

        from mimir.llm import router

        source = _inspect.getsource(router.ModelRouter.constrained)
        assert "max_output_tokens" in source

    def test_the_loop_retries_once_on_a_length_stop(self):
        import inspect as _inspect

        from mimir.agent.loop import AgentLoop

        source = _inspect.getsource(AgentLoop._constrained_step)
        assert 'reason == "length"' in source
        assert "16_384" in source

    def test_a_second_truncation_is_an_error_not_an_answer(self):
        import inspect as _inspect

        from mimir.agent.loop import AgentLoop

        source = _inspect.getsource(AgentLoop._constrained_step)
        assert "cut off mid-call" in source
