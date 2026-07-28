import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamInvestigation } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { ApprovalCard } from "./components/ApprovalCard";
import { EvidenceList } from "./components/EvidenceList";
import { Timeline } from "./components/Timeline";
import { Empty, Panel } from "./components/primitives";
import type {
  Evidence,
  FinalAnswer,
  Hypothesis,
  PendingApproval,
  RunEvent,
  SessionSummary,
} from "./types";

type Tab = "answer" | "evidence" | "timeline";

export default function App() {
  const [question, setQuestion] = useState("");
  const [clusterContext, setClusterContext] = useState("");
  const [namespace, setNamespace] = useState("");
  const [repository, setRepository] = useState("");

  const [events, setEvents] = useState<RunEvent[]>([]);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [answer, setAnswer] = useState<FinalAnswer | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("timeline");

  const [approvals, setApprovals] = useState<PendingApproval[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const abortRef = useRef<(() => void) | null>(null);

  const refreshApprovals = useCallback(async () => {
    try {
      const result = await api.approvals();
      setApprovals(result.approvals);
    } catch {
      // The API may be down; the banner below already reports run failures.
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      const result = await api.sessions();
      setSessions(result.sessions.slice(0, 15));
    } catch {
      /* non-fatal */
    }
  }, []);

  // Poll for approvals while a run is in flight. An approval can be raised deep
  // inside a tool call, and the operator may be looking at another tab.
  useEffect(() => {
    void refreshApprovals();
    void refreshSessions();
    if (!running) return;
    const timer = setInterval(() => void refreshApprovals(), 1500);
    return () => clearInterval(timer);
  }, [running, refreshApprovals, refreshSessions]);

  useEffect(() => () => abortRef.current?.(), []);

  function start() {
    if (!question.trim() || running) return;
    abortRef.current?.();
    setEvents([]);
    setEvidence([]);
    setAnswer(null);
    setHypotheses([]);
    setConfidence(0);
    setError(null);
    setRunning(true);
    setTab("timeline");

    abortRef.current = streamInvestigation(
      {
        question,
        cluster_context: clusterContext || undefined,
        namespace: namespace || undefined,
        repositories: repository ? [repository] : undefined,
      },
      (event) => {
        setEvents((prev) => [...prev, event]);
        if (event.type === "started") setSessionId(String(event.session_id));
        if (event.type === "evidence") {
          setEvidence((prev) => {
            const item = event as unknown as Evidence;
            return prev.some((e) => e.id === item.id) ? prev : [...prev, item];
          });
        }
        if (event.type === "approval") void refreshApprovals();
        if (event.type === "answer") {
          setAnswer(event.answer as FinalAnswer);
          setConfidence(Number(event.confidence ?? 0));
          setTab("answer");
        }
        if (event.type === "error") setError(String(event.error));
        if (event.type === "done") {
          setRunning(false);
          void refreshSessions();
          void loadSessionDetail(String(event.session_id));
        }
      },
      (message) => {
        setError(message);
        setRunning(false);
      },
    );
  }

  async function loadSessionDetail(id: string) {
    try {
      const state = (await api.session(id)) as Record<string, unknown>;
      setHypotheses((state.hypotheses as Hypothesis[]) ?? []);
      const rejected = (state.rejected_hypotheses as Hypothesis[]) ?? [];
      if (rejected.length) setHypotheses((prev) => [...prev, ...rejected]);
      setEvidence((state.evidence as Evidence[]) ?? []);
    } catch {
      /* the streamed data is already displayed */
    }
  }

  async function openSession(id: string) {
    try {
      const state = (await api.session(id)) as Record<string, unknown>;
      setSessionId(id);
      setQuestion(String(state.user_request ?? ""));
      setEvidence((state.evidence as Evidence[]) ?? []);
      setAnswer((state.final_answer as FinalAnswer) ?? null);
      setConfidence(Number(state.final_confidence ?? 0));
      setHypotheses([
        ...((state.hypotheses as Hypothesis[]) ?? []),
        ...((state.rejected_hypotheses as Hypothesis[]) ?? []),
      ]);
      setEvents([]);
      setTab("answer");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function exportSession() {
    if (!sessionId) return;
    try {
      const result = await api.exportSession(sessionId, "md");
      const blob = new Blob([result.content], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${sessionId}.md`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-[1400px] flex-col gap-4 p-4">
      <header className="flex items-baseline justify-between">
        <h1 className="text-lg font-semibold tracking-tight text-neutral-100">MIMIR</h1>
        <p className="text-xs text-neutral-600">
          local operations investigation - read-only by default, mutations need approval
        </p>
      </header>

      <Panel title="new investigation">
        <div className="space-y-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) start();
            }}
            rows={2}
            placeholder="Why is the checkout service timing out against auth?"
            className="w-full resize-y rounded border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm outline-none focus:border-neutral-600"
          />
          <div className="flex flex-wrap gap-2">
            <Field value={clusterContext} onChange={setClusterContext} placeholder="cluster context" />
            <Field value={namespace} onChange={setNamespace} placeholder="namespace" />
            <Field value={repository} onChange={setRepository} placeholder="repository" />
            <button
              onClick={start}
              disabled={running || !question.trim()}
              className="rounded bg-neutral-100 px-4 py-1.5 text-sm font-medium text-neutral-900 hover:bg-white disabled:opacity-40"
            >
              {running ? "running..." : "investigate"}
            </button>
            {running ? (
              <button
                onClick={() => {
                  abortRef.current?.();
                  setRunning(false);
                }}
                className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
              >
                stop
              </button>
            ) : null}
            {sessionId && !running ? (
              <button
                onClick={exportSession}
                className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
              >
                export evidence package
              </button>
            ) : null}
          </div>
        </div>
      </Panel>

      {error ? (
        <div className="rounded border border-red-800 bg-red-950/40 p-3 text-sm text-red-300">
          {error}
        </div>
      ) : null}

      {approvals.length ? (
        <Panel title={`pending approvals (${approvals.length})`}>
          <div className="space-y-3">
            {approvals.map((approval) => (
              <ApprovalCard
                key={approval.id}
                approval={approval}
                onResolved={refreshApprovals}
              />
            ))}
          </div>
        </Panel>
      ) : null}

      <div className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-[1fr,300px]">
        <Panel
          title={
            <div className="flex gap-3">
              {(["answer", "evidence", "timeline"] as Tab[]).map((name) => (
                <button
                  key={name}
                  onClick={() => setTab(name)}
                  className={
                    tab === name ? "text-neutral-100" : "text-neutral-600 hover:text-neutral-400"
                  }
                >
                  {name}
                  {name === "evidence" && evidence.length ? ` (${evidence.length})` : ""}
                </button>
              ))}
            </div>
          }
        >
          {tab === "answer" ? (
            <AnswerPanel answer={answer} confidence={confidence} hypotheses={hypotheses} />
          ) : null}
          {tab === "evidence" ? <EvidenceList items={evidence} /> : null}
          {tab === "timeline" ? <Timeline events={events} /> : null}
        </Panel>

        <Panel title="recent sessions">
          {sessions.length ? (
            <ul className="space-y-1">
              {sessions.map((session) => (
                <li key={session.id}>
                  <button
                    onClick={() => void openSession(session.id)}
                    className="w-full truncate text-left text-xs text-neutral-400 hover:text-neutral-100"
                    title={session.user_request}
                  >
                    {session.user_request || session.id}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No sessions yet.</Empty>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Field({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-40 rounded border border-neutral-800 bg-neutral-950 px-2 py-1.5 text-xs outline-none focus:border-neutral-600"
    />
  );
}
