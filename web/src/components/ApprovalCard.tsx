import { useState } from "react";
import { api } from "../api";
import type { PendingApproval } from "../types";
import { RiskBadge } from "./primitives";

/**
 * The ADR 13.3 pre-execution display, rendered before a human decides.
 * Everything the ADR requires is shown: exact command, resolved target context,
 * expected effect, risk class with reasons, and the rollback plan.
 */
export function ApprovalCard({
  approval,
  onResolved,
}: {
  approval: PendingApproval;
  onResolved: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState(approval.command);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(decision: string) {
    setBusy(true);
    setError(null);
    try {
      const extra =
        decision === "edit" ? { edited_argv: splitArgv(edited) } : {};
      await api.decide(approval.id, decision, extra);
      onResolved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const contextRows = Object.entries(approval.context ?? {}).filter(
    ([, value]) => value !== null && value !== undefined && value !== "",
  );

  return (
    <div
      className={`rounded-lg border p-3 ${
        approval.risk === "R4" || approval.production_target
          ? "border-red-700 bg-red-950/30"
          : "border-amber-800 bg-amber-950/20"
      }`}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="text-xs uppercase tracking-wide text-neutral-400">
          approval required
        </span>
        <RiskBadge risk={approval.risk} />
      </div>

      {editing ? (
        <input
          value={edited}
          onChange={(e) => setEdited(e.target.value)}
          className="mono w-full rounded border border-neutral-700 bg-neutral-950 px-2 py-1.5"
          spellCheck={false}
        />
      ) : (
        <pre className="mono overflow-x-auto rounded bg-neutral-950 p-2 text-neutral-100">
          $ {approval.command}
        </pre>
      )}

      <dl className="mt-2 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs">
        {contextRows.map(([key, value]) => (
          <Row key={key} label={key.replace(/_/g, " ")} value={String(value)} />
        ))}
        {approval.purpose ? <Row label="reason" value={approval.purpose} /> : null}
        {approval.expected_effect ? (
          <Row label="expected effect" value={approval.expected_effect} />
        ) : null}
        <Row
          label="rollback"
          value={approval.rollback_hint ?? "none available; verify manually"}
        />
      </dl>

      {approval.reasons.length ? (
        <ul className="mt-2 space-y-0.5 text-xs text-neutral-400">
          {approval.reasons.map((reason) => (
            <li key={reason}>- {reason}</li>
          ))}
        </ul>
      ) : null}

      {approval.production_target ? (
        <p className="mt-2 text-xs font-medium text-red-400">
          This target matches a production pattern. Read the command again.
        </p>
      ) : null}
      {!approval.reversible ? (
        <p className="mt-1 text-xs font-medium text-red-400">
          This action is not automatically reversible.
        </p>
      ) : null}

      {error ? <p className="mt-2 text-xs text-red-400">{error}</p> : null}

      <div className="mt-3 flex flex-wrap gap-2">
        <button
          disabled={busy}
          onClick={() => decide(editing ? "edit" : "approve")}
          className="rounded bg-emerald-700 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
        >
          {editing ? "approve edited" : "approve"}
        </button>
        <button
          disabled={busy}
          onClick={() => decide("reject")}
          className="rounded bg-red-800 px-3 py-1 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-50"
        >
          reject
        </button>
        <button
          disabled={busy}
          onClick={() => setEditing((v) => !v)}
          className="rounded border border-neutral-700 px-3 py-1 text-xs text-neutral-300 hover:bg-neutral-800 disabled:opacity-50"
        >
          {editing ? "cancel edit" : "edit"}
        </button>
      </div>
      {editing ? (
        <p className="mt-2 text-[11px] text-neutral-500">
          An edited command is re-classified from scratch. If it turns out riskier than
          the original it will be refused.
        </p>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-neutral-500">{label}</dt>
      <dd className="mono text-neutral-300">{value}</dd>
    </>
  );
}

/** Minimal argv split. The backend re-parses and re-validates regardless. */
function splitArgv(input: string): string[] {
  const out: string[] = [];
  const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(input)) !== null) {
    out.push(match[1] ?? match[2] ?? match[3]);
  }
  return out;
}
