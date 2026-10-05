"""construct_command answers from the parser when everything is stated."""

from mimir.agent.command import construct_fast
from mimir.safety.policy import get_policy_engine


def test_logs_with_namespace_and_tail_is_built_without_a_model():
    (cmd,) = construct_fast("show the last 50 log lines of deployment api in namespace payments --context ch1")
    assert cmd.argv[:1] == ["kubectl"]
    assert "-n" in cmd.argv and cmd.argv[cmd.argv.index("-n") + 1] == "payments"
    assert "logs" in cmd.argv and "deployment/api" in cmd.argv and "50" in cmd.argv
    assert cmd.context.namespace == "payments"


def test_an_unstated_namespace_returns_none_so_the_graph_or_the_store_decides():
    assert construct_fast("get logs for the api deployment") is None


def test_the_entity_store_can_supply_a_single_known_namespace(tmp_path):
    from mimir.knowledge.entities import EntityStore

    class R:
        ok = True
        data = {"context": "ch1", "namespace": "payments", "workloads": [{"kind": "Deployment", "name": "api"}]}

    store = EntityStore(tmp_path / "e.db"); store.observe("list_workloads", None, R())
    (cmd,) = construct_fast("describe the api deployment", entities=store)
    assert cmd.context.namespace == "payments" and cmd.context.cluster_context == "ch1"


def test_two_known_namespaces_is_not_a_guess(tmp_path):
    from mimir.knowledge.entities import EntityStore

    class R:
        ok = True
        def __init__(self, ns): self.data = {"context": "ch1", "namespace": ns, "workloads": [{"kind": "Deployment", "name": "api"}]}

    store = EntityStore(tmp_path / "e.db"); store.observe("w", None, R("payments")); store.observe("w", None, R("staging"))
    assert construct_fast("describe the api deployment", entities=store) is None


def test_every_fast_command_is_read_only_and_classified_r0_or_r1(settings):
    engine = get_policy_engine(settings)
    # Every phrasing names the kind: a bare word is not a stated name and the parser declines it (next test), which 
    for text in ("events for the api deployment in namespace payments",
                 "describe pod api-7c9 in namespace payments",
                 "status of deployment api in namespace payments",
                 "cpu usage of the api pods in namespace payments",
                 "restarts of deployment api in namespace payments",
                 "logs of api pod in payments namespace since 10m"):
        (cmd,) = construct_fast(text)
        assert str(engine.classify_only(cmd).risk).upper().endswith(("R0", "R1")), (text, cmd.argv)


def test_no_action_means_the_parser_declines():
    assert construct_fast("api in namespace payments") is None


def test_a_bare_word_with_no_kind_noun_is_not_a_stated_name():
    assert construct_fast("restarts of api in namespace payments") is None


def test_a_stated_cluster_fragment_is_resolved_or_the_parser_declines(tmp_path):
    """"on cluster ch1" must reach --context or stop the fast path; it was silently dropped once."""
    from mimir.knowledge.entities import EntityStore

    class R:
        ok = True
        data = {"context": "gce-management-ch1-dev", "available_contexts": ["gce-management-ch1-dev", "gce-prod-eu-1"]}

    text = "show the last 50 log lines of deployment api in namespace payments on cluster ch1"
    assert construct_fast(text) is None
    store = EntityStore(tmp_path / "e.db"); store.observe("get_current_context", None, R())
    (cmd,) = construct_fast(text, entities=store)
    assert cmd.argv[cmd.argv.index("--context") + 1] == "gce-management-ch1-dev"
