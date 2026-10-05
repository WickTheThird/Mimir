"""The entity store: what was seen, and targeting from it."""

import time

import pytest

from mimir.decide.base import Verdict
from mimir.graph.nodes import resolve_target
from mimir.knowledge.entities import EntityStore, entity_id
from mimir.models.state import InvestigationState


class R:
    def __init__(self, data, ok=True):
        self.data, self.ok = data, ok


@pytest.fixture
def store(tmp_path):
    return EntityStore(tmp_path / "entities.db")


def test_a_workload_listing_becomes_entities_and_edges(store):
    n = store.observe("list_workloads", None, R({
        "context": "prod-eu-1", "namespace": "payments",
        "workloads": [{"kind": "Deployment", "name": "api", "pods": ["api-7c9", "api-8d1"], "desired": 2}],
    }))
    assert n >= 3
    api = store.get(entity_id("deployment", "api", "payments", "prod-eu-1"))
    assert api is not None and api.attrs["desired"] == 2
    chain = store.walk(entity_id("pod", "api-7c9", "payments", "prod-eu-1"), "owned_by")
    assert [e.name for e in chain] == ["api", "payments", "prod-eu-1"]


def test_find_workloads_matches_are_read_the_same_way(store):
    store.observe("find_workloads", None, R({"matches": [
        {"kind": "Deployment", "name": "messaging-whatsapp", "namespace": "messaging-squad", "context": "ch1"}
    ]}))
    (hit,) = store.candidates("whatsapp")
    assert hit.scope == "ch1/messaging-squad"


def test_a_failed_result_is_not_an_observation(store):
    assert store.observe("list_workloads", None, R({"workloads": [{"name": "x"}]}, ok=False)) == 0


def test_an_unreadable_result_never_raises(store):
    assert store.observe("list_workloads", None, R({"workloads": "not a list"})) == 0


def test_stale_facts_are_findable_by_age(store):
    store.observe("list_workloads", None, R({"context": "c", "namespace": "n",
                                             "workloads": [{"name": "old"}]}))
    store._db.execute("UPDATE entities SET seen_at=?", (time.time() - 40 * 86400,)); store.commit()
    assert {e.name for e in store.stale(30 * 86400)} >= {"old"}


class FakeDecider:
    available = True
    name = "fake"

    def __init__(self, choice):
        self.choice = choice

    def decide(self, context, fields):
        f = fields[0]
        others = [o for o in f.options if o != self.choice]
        dist = {self.choice: 0.9, **{o: 0.1 / len(others) for o in others}}
        return {f.name: Verdict(field=f.name, choice=self.choice, probability=0.9, distribution=dist)}


def _seen(store, *scopes):
    for ctx, ns in scopes:
        store.observe("list_workloads", None, R({"context": ctx, "namespace": ns,
                                                 "workloads": [{"kind": "Deployment", "name": "api"}]}))


@pytest.mark.asyncio
async def test_one_known_place_binds_without_a_model(store):
    _seen(store, ("prod-eu-1", "payments"))
    session = InvestigationState(user_request="restart the api deployment")
    out = await resolve_target(session, store, FakeDecider("ask the operator"))
    assert out["status"] == "bound" and out["source"] == "store"
    assert session.environment.namespace == "payments"
    assert session.environment.cluster_context == "prod-eu-1"


@pytest.mark.asyncio
async def test_several_places_and_a_confident_pick_binds(store):
    _seen(store, ("prod-eu-1", "payments"), ("prod-eu-1", "staging"))
    session = InvestigationState(user_request="restart the api deployment in payments")
    out = await resolve_target(session, store, FakeDecider("prod-eu-1/payments"))
    assert out["status"] == "bound" and out["source"] == "decider"
    assert session.environment.namespace == "payments"
    assert session.metadata["decisions"][0]["field"] == "target"


@pytest.mark.asyncio
async def test_several_places_and_no_basis_to_pick_asks(store):
    """Guessing costs a command on the wrong namespace."""
    _seen(store, ("prod-eu-1", "payments"), ("prod-eu-1", "staging"))
    session = InvestigationState(user_request="restart the api deployment")
    out = await resolve_target(session, store, FakeDecider("ask the operator"))
    assert out["status"] == "ambiguous"
    assert session.pending_questions and "payments" in session.pending_questions[0]
    assert session.environment.namespace is None


@pytest.mark.asyncio
async def test_the_operator_scope_is_respected(store):
    _seen(store, ("prod-eu-1", "payments"), ("prod-eu-1", "staging"))
    session = InvestigationState(user_request="restart the api deployment")
    session.environment.namespace = "staging"; session.environment.cluster_context = "prod-eu-1"
    out = await resolve_target(session, store, FakeDecider("prod-eu-1/payments"))
    assert out["status"] == "scoped_by_operator"
    assert session.environment.namespace == "staging"


@pytest.mark.asyncio
async def test_nothing_seen_leaves_it_to_the_plan(store):
    session = InvestigationState(user_request="restart the api deployment")
    assert (await resolve_target(session, store, FakeDecider("x")))["status"] == "unknown"


@pytest.mark.asyncio
async def test_no_name_means_nothing_to_resolve(store):
    session = InvestigationState(user_request="why is checkout timing out")
    assert (await resolve_target(session, store, None))["status"] == "unnamed"


def test_find_workloads_rows_keyed_pod_are_recorded(store):
    n = store.observe("find_workloads", None, R({"matches": [
        {"context": "gce-backend-fr5-prod", "namespace": "messaging-squad", "pod": "messaging-settings-65f5-mhfdl", "phase": "Running"}]}))
    assert n >= 1
    (hit,) = store.candidates("messaging-settings")
    assert hit.kind == "pod" and hit.scope == "gce-backend-fr5-prod/messaging-squad"
