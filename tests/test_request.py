"""Deterministic extraction of what the operator asked for.

Every case is a phrasing from a real session. The governing rule is that this
never guesses: a parser that fills in a plausible value when it is unsure
reintroduces the silent wrongness it exists to remove.
"""

from __future__ import annotations

from mimir.agent.request import parse_request


class TestTheRequestThatStartedThis:
    def test_every_parameter_is_taken_out_of_the_sentence(self):
        """The council was given five parameters in one sentence, carried none
        of them through six tool calls, and reported that a pod which was
        running did not exist."""
        request = parse_request(
            "can you tell me the last 10 logs of any messaging outbound pod "
            "that is inside dev and in a cluster with ch1?"
        )
        assert request.name_contains == "messaging-outbound"
        assert request.context_contains == "ch1"
        assert request.environment == "dev"
        assert request.action == "logs"
        assert request.tail == 10
        assert not request.namespace, "none was stated, so none is invented"

    def test_a_widening_fallback_is_offered_but_not_taken(self):
        """"messaging outbound" widens to "outbound", not to "messaging",
        which is a prefix of forty other things."""
        request = parse_request("logs of any messaging outbound pod")
        assert request.name_candidates[0] == "messaging-outbound"
        assert "outbound" in request.name_candidates
        assert "messaging" not in request.name_candidates


class TestScope:
    def test_a_namespace_before_the_noun_beats_the_word_after_it(self):
        """"in the payments namespace over the last 2 hours" matches
        "namespace over" if the forms are tried in the wrong order."""
        request = parse_request("show restarts in the payments namespace over the last 2 hours")
        assert request.namespace == "payments"

    def test_a_flag_is_read(self):
        assert parse_request("-n messaging-squad get logs").namespace == "messaging-squad"

    def test_a_cluster_described_two_words_out_is_still_found(self):
        """"the ch1 dev cluster" puts the environment nearest the noun, so
        taking only the adjacent word yields "dev", which names no cluster."""
        request = parse_request("tail 50 lines from the kannel client pods on the ch1 dev cluster")
        assert request.context_contains == "ch1"
        assert request.environment == "dev"
        assert request.name_contains == "kannel-client"

    def test_a_named_context_is_not_a_fragment(self):
        request = parse_request("logs for api with context prod-eu-1")
        assert request.context == "prod-eu-1"
        assert not request.context_contains


class TestCountsAndWindows:
    def test_a_count_is_a_count(self):
        assert parse_request("the last 10 logs of api pods").tail == 10
        assert parse_request("tail 50 lines from api").tail == 50

    def test_a_duration_is_not_a_count(self):
        """"the last 2 hours" read as two lines silently answers a different
        question."""
        request = parse_request("show restarts over the last 2 hours")
        assert request.tail == 0
        assert request.since == "2h"

    def test_minutes_are_recognised(self):
        assert parse_request("fetch the last 30 minutes of logs for api").since == "30m"


class TestNames:
    def test_kind_slash_name_wins_over_prose(self):
        request = parse_request("get the logs for deployment/api in namespace payments")
        assert request.name_contains == "api"
        assert request.namespace == "payments"

    def test_an_environment_is_never_a_workload_name(self):
        assert parse_request("logs from the dev pods").name_contains != "dev"

    def test_a_question_with_no_target_states_nothing(self):
        assert parse_request("why is checkout slow").stated == {}


class TestRouting:
    def _kind(self, text):
        from mimir.graph.triage import triage

        return triage(text).kind.value

    def test_a_described_target_routes_to_the_loop(self):
        """The first version required a namespace flag or a hyphenated name, so
        this fell through to the council, which listed 211 namespaces, invented
        a pod to exec into, never read a log line, and reported that no such
        pod existed. Two were running."""
        assert self._kind(
            "can you tell me the last 10 logs of any messaging outbound pod "
            "that is inside dev and in a cluster with ch1?"
        ) == "direct"

    def test_a_mutation_never_reaches_the_read_only_loop(self):
        """"restart the api deployment" parses as an action against a named
        target, and the loop reads. Mutation belongs to the council, which has
        the prepare, approve and execute path."""
        for text in (
            "restart the api deployment",
            "scale the api deployment to 5 in namespace payments",
            "Raise the memory limit on the api deployment in the prod-eu-1 cluster.",
            "delete the failing pod in -n payments",
        ):
            assert self._kind(text) == "investigate", text

    def test_direct_never_short_circuits_the_graph(self):
        """Marking it cheap returned a canned reply that does not exist for
        this verdict, and would have changed what the corpus measures."""
        from mimir.graph.triage import triage

        verdict = triage("fetch the last 30 minutes of logs for the api deployment")
        assert verdict.kind.value == "direct"
        assert not verdict.cheap


class TestBinding:
    """What the parser found is applied, not suggested."""

    def _agent(self):
        from mimir.agent.ops import OpsAgent
        from mimir.tools.base import ToolContext, load_all_tools

        class Router:
            call_log: list = []
            invocations_attempted = 0

            def for_task(self, task):
                return None

            def digest_for(self, alias):
                return ""

        registry = load_all_tools()
        return OpsAgent(
            router=Router(),
            registry=registry,
            tool_context=ToolContext(registry=registry),
        )

    def test_the_environment_is_part_of_the_cluster_constraint(self):
        """Left out, "any outbound pod in dev in a cluster with ch1" returned
        the ch1 production clusters too."""
        agent = self._agent()
        agent.note_instruction(
            "logs of any messaging outbound pod inside dev in a cluster with ch1"
        )
        bound = agent.bind("find_workloads", {})
        assert bound["context_contains"] == "ch1 dev"
        assert bound["name_contains"] == "messaging-outbound"

    def test_a_namespace_nobody_named_is_cleared_not_defaulted(self):
        """The model read "dev" as a namespace. No namespace is called dev, so
        a search that would have found both pods returned nothing, and the
        emptiness looked like an answer."""
        agent = self._agent()
        agent.note_instruction("logs of any outbound pod inside dev with ch1")
        assert agent.bind("find_workloads", {"namespace_contains": "dev"})[
            "namespace_contains"
        ] is None

    def test_a_namespace_the_operator_named_is_applied(self):
        agent = self._agent()
        agent.note_instruction("logs of outbound pods in -n messaging-squad")
        assert agent.bind("find_workloads", {})["namespace_contains"] == "messaging-squad"

    def test_a_stated_line_count_is_not_negotiable(self):
        """"the last 10 logs" returning a hundred lines has answered a
        different question."""
        agent = self._agent()
        agent.note_instruction("the last 10 logs of api pods in -n payments")
        assert agent.bind("get_logs", {"target": "api", "tail": 500})["tail"] == 10


class TestALocatedPodCarriesItsCluster:
    """A search that spans clusters returns the context each match lives in,
    and the next call names the pod without it. Left alone the pod name
    resolves against whatever the kubeconfig points at: a request for logs from
    a ch1 dev cluster returned logs from an unrelated one, and the answer named
    the wrong cluster while looking entirely correct."""

    def _agent(self):
        from mimir.agent.ops import OpsAgent
        from mimir.tools.base import ToolContext, load_all_tools

        class Router:
            call_log: list = []
            invocations_attempted = 0

            def for_task(self, task):
                return None

            def digest_for(self, alias):
                return ""

        registry = load_all_tools()
        agent = OpsAgent(
            router=Router(),
            registry=registry,
            tool_context=ToolContext(registry=registry),
        )
        agent.note_instruction("logs of any outbound pod in dev in a cluster with ch1")
        return agent

    def _found(self, agent, context="aws-backend-ch1-dev"):
        from mimir.tools.base import ToolResult

        agent.note_success("find_workloads", ToolResult(
            tool="find_workloads",
            data={"matches": [{"context": context, "namespace": "messaging-squad",
                               "pod": "messaging-outbound-abc"}]},
        ))

    def test_reading_a_located_pod_uses_the_cluster_it_was_found_in(self):
        agent = self._agent()
        self._found(agent)
        bound = agent.bind("get_logs", {"target": "messaging-outbound-abc", "tail": 10})
        assert bound["context"] == "aws-backend-ch1-dev"
        assert bound["namespace"] == "messaging-squad"

    def test_a_context_the_call_states_still_wins(self):
        agent = self._agent()
        self._found(agent)
        bound = agent.bind("get_logs", {"target": "messaging-outbound-abc",
                                        "context": "somewhere-else"})
        assert bound["context"] == "somewhere-else"

    def test_a_pod_that_was_never_located_is_not_given_a_cluster(self):
        agent = self._agent()
        self._found(agent)
        bound = agent.bind("get_logs", {"target": "some-other-pod"})
        assert bound.get("context") in (None, "")

    def test_a_turn_does_not_inherit_the_last_turn_s_pods(self):
        agent = self._agent()
        self._found(agent)
        agent.note_instruction("now check events")
        assert agent.located == {}


class TestTheTurnEndsWhenTheAskedActionIsDone:
    """Under a constrained decoder the answer branch is always available and
    the model does not reliably take it. A run that retrieved exactly the
    requested log lines then fetched them another five times."""

    def _agent(self):
        return TestALocatedPodCarriesItsCluster._agent(self)

    def test_tools_stay_available_until_the_action_succeeds(self):
        agent = self._agent()
        assert agent.specs_now()

    def test_once_the_action_has_succeeded_only_answering_is_left(self):
        from mimir.tools.base import ToolResult

        agent = self._agent()
        agent.note_success("get_logs", ToolResult(tool="get_logs"))
        assert agent.specs_now() == []

    def test_an_unrelated_success_does_not_end_the_turn(self):
        from mimir.tools.base import ToolResult

        agent = self._agent()
        agent.note_success("list_workloads", ToolResult(tool="list_workloads"))
        assert agent.specs_now()


class TestALocatedPodOverridesAGuess:
    """A smaller model guesses the scope worse. One put the context and the
    namespace into the namespace field as a single slash-joined string, four
    times over, and a fill-if-absent rule let the wrong value stand because the
    field was not empty."""

    def _agent_with(self, context="aws-backend-ch1-dev", namespace="messaging-squad"):
        from mimir.tools.base import ToolResult

        agent = TestALocatedPodCarriesItsCluster._agent(self)
        agent.note_success("find_workloads", ToolResult(
            tool="find_workloads",
            data={"matches": [{"context": context, "namespace": namespace,
                               "pod": "messaging-outbound-abc"}]},
        ))
        return agent

    def test_a_wrong_namespace_is_replaced_by_where_it_was_found(self):
        agent = self._agent_with()
        bound = agent.bind("get_logs", {
            "target": "messaging-outbound-abc",
            "namespace": "aws-backend-ch1-dev/messaging-squad",
        })
        assert bound["namespace"] == "messaging-squad"
        assert bound["context"] == "aws-backend-ch1-dev"

    def test_a_wrong_context_is_replaced_too(self):
        agent = self._agent_with()
        bound = agent.bind("get_logs", {
            "target": "messaging-outbound-abc", "context": "somewhere-else",
        })
        assert bound["context"] == "aws-backend-ch1-dev"

    def test_a_pod_that_was_never_located_keeps_what_the_call_said(self):
        """Only an observation beats a guess. Without one, the guess stands."""
        agent = self._agent_with()
        bound = agent.bind("get_logs", {"target": "other-pod", "context": "ctx-x"})
        assert bound["context"] == "ctx-x"
