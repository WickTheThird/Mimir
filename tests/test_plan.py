"""Multi-step tasks: checkpoint what lands, roll back what fails."""

import subprocess

import pytest

from mimir.agent.plan import Step, TaskPlan, checkpoint, plan_steps, rollback, run_plan


class Ev:
    def __init__(self, text=""):
        from mimir.agent.loop import AgentEventType
        self.type = AgentEventType.TEXT
        self.text = text


class Outcome:
    def __init__(self, stopped="done", files=()):
        self.stopped, self.files_changed = stopped, set(files)


class ScriptedAgent:
    """Each run writes a file (or not) and reports an outcome."""

    def __init__(self, root, script):
        self.root, self.script, self.instructions = root, list(script), []

    async def run(self, instruction):
        self.instructions.append(instruction)
        name, ok = self.script.pop(0)
        if name:
            (self.root / name).write_text("x")
        self.outcome = Outcome("done" if ok else "error", [name] if name else [])
        yield Ev("did it")


@pytest.fixture
def root(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    checkpoint(tmp_path, "init")
    return tmp_path


@pytest.mark.asyncio
async def test_each_landed_step_is_a_checkpoint_and_later_steps_see_earlier_ones(root):
    agent = ScriptedAgent(root, [("b.txt", True), ("c.txt", True)])
    plan = TaskPlan("two things", [Step("add b", "add b"), Step("add c", "add c")])
    plan = await run_plan(agent, plan, worktree_root=root)
    assert [s.status for s in plan.steps] == ["done", "done"]
    assert all(s.checkpoint for s in plan.steps)
    assert "Steps already done and committed: add b" in agent.instructions[1]
    log = subprocess.run(["git", "-C", str(root), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "checkpoint: add c" in log


@pytest.mark.asyncio
async def test_a_failed_step_is_rolled_back_and_later_steps_skipped(root):
    agent = ScriptedAgent(root, [("b.txt", True), ("bad.txt", False), ("c.txt", True)])
    plan = TaskPlan("three", [Step("b", "b"), Step("bad", "bad"), Step("c", "c")])
    plan = await run_plan(agent, plan, worktree_root=root)
    assert [s.status for s in plan.steps] == ["done", "failed", "skipped"]
    assert (root / "b.txt").exists() and not (root / "bad.txt").exists()
    assert len(agent.instructions) == 2


@pytest.mark.asyncio
async def test_a_custom_gate_decides_what_may_be_kept(root):
    agent = ScriptedAgent(root, [("b.txt", True)])
    plan = TaskPlan("one", [Step("b", "b")])
    plan = await run_plan(agent, plan, worktree_root=root, gate_ok=lambda a: False)
    assert plan.steps[0].status == "failed" and not (root / "b.txt").exists()


@pytest.mark.asyncio
async def test_a_plan_that_cannot_be_made_is_one_step():
    class Broken:
        async def chat(self, *a, **k):
            from mimir.llm.base import ModelError
            raise ModelError("down")

    plan = await plan_steps(Broken(), "do the thing")
    assert [s.instruction for s in plan.steps] == ["do the thing"]


@pytest.mark.asyncio
async def test_plan_steps_reads_the_structured_reply():
    class R:
        async def chat(self, *a, **k):
            class Resp:
                content = '{"steps": [{"title": "model", "instruction": "add the model", "done_when": "it imports"}, {"title": "endpoint", "instruction": "add the endpoint", "done_when": "200"}]}'
            return Resp()

    plan = await plan_steps(R(), "add model and endpoint")
    assert [s.title for s in plan.steps] == ["model", "endpoint"]
    assert plan.steps[0].done_when == "it imports"
