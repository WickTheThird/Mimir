"""Repository against cluster: equalities, not opinions."""

from mimir.knowledge.entities import EntityStore
from mimir.verify.manifests import compare, declared_workloads


class R:
    def __init__(self, data):
        self.data, self.ok = data, True


def test_manifests_are_read_from_yaml_including_multi_document(tmp_path):
    (tmp_path / "k8s").mkdir()
    (tmp_path / "k8s" / "api.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: api\n  namespace: payments\n"
        "spec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n      - name: api\n        image: repo/api:1.2\n"
        "---\napiVersion: v1\nkind: Service\nmetadata:\n  name: api\n"
    )
    (d,) = declared_workloads(tmp_path)
    assert (d.kind, d.name, d.namespace, d.replicas, d.images) == ("Deployment", "api", "payments", 3, ["repo/api:1.2"])


def test_drift_is_a_difference_in_replicas_or_images(tmp_path):
    (tmp_path / "api.yaml").write_text(
        "kind: Deployment\nmetadata: {name: api, namespace: payments}\nspec:\n  replicas: 3\n"
        "  template: {spec: {containers: [{name: api, image: repo/api:1.2}]}}\n"
    )
    store = EntityStore(tmp_path / "e.db")
    store.observe("list_workloads", None, R({"context": "prod", "namespace": "payments", "workloads": [
        {"kind": "Deployment", "name": "api", "desired": 6, "images": ["repo/api:1.1"]}]}))
    drifts = compare(declared_workloads(tmp_path), store)
    assert sorted((d.field, str(d.declared), str(d.observed)) for d in drifts) == [("images", "['repo/api:1.2']", "['repo/api:1.1']"), ("replicas", "3", "6")]
    assert "repository says 3" in drifts[0].render()


def test_agreement_and_unseen_workloads_produce_no_drift(tmp_path):
    (tmp_path / "api.yaml").write_text("kind: Deployment\nmetadata: {name: api, namespace: payments}\nspec: {replicas: 2}\n")
    (tmp_path / "other.yaml").write_text("kind: Deployment\nmetadata: {name: other}\nspec: {replicas: 1}\n")
    store = EntityStore(tmp_path / "e.db")
    store.observe("list_workloads", None, R({"context": "prod", "namespace": "payments", "workloads": [
        {"kind": "Deployment", "name": "api", "desired": 2}]}))
    assert compare(declared_workloads(tmp_path), store) == []
