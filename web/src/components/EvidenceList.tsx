import { useState } from "react";
import { api } from "../api";
import type { Evidence } from "../types";
import { Empty } from "./primitives";

const KIND_STYLE: Record<string, string> = {
  observed: "text-emerald-400",
  inferred: "text-amber-400",
  hypothesis: "text-orange-400",
};

const FRESHNESS_STYLE: Record<string, string> = {
  live: "text-emerald-500",
  recent: "text-neutral-500",
  stale: "text-orange-400",
  unknown: "text-neutral-600",
};

/**
 * Evidence with file and line citations (ADR 15). Contradicting evidence is
 * rendered distinctly rather than hidden, because ADR 7.2 requires that
 * disagreement stays visible.
 */
export function EvidenceList({ items }: { items: Evidence[] }) {
  if (!items.length) return <Empty>No evidence gathered yet.</Empty>;
  return (
    <ul className="space-y-2">
      {items.map((item) => (
        <EvidenceRow key={item.id} item={item} />
      ))}
    </ul>
  );
}

function EvidenceRow({ item }: { item: Evidence }) {
  const [open, setOpen] = useState(false);
  const [artifact, setArtifact] = useState<string | null>(null);

  async function loadArtifact() {
    if (!item.artifact_ref || artifact) return;
    try {
      const result = await api.artifact(item.artifact_ref);
      setArtifact(result.content);
    } catch (e) {
      setArtifact(`failed to load: ${(e as Error).message}`);
    }
  }

  return (
    <li
      className={`rounded border p-2 ${
        item.supports ? "border-neutral-800" : "border-red-900 bg-red-950/20"
      }`}
    >
      <button
        onClick={() => {
          setOpen((v) => !v);
          void loadArtifact();
        }}
        className="w-full text-left"
      >
        <div className="flex items-start gap-2">
          <span className={item.supports ? "text-emerald-500" : "text-red-400"}>
            {item.supports ? "+" : "!"}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm text-neutral-200">{item.claim}</p>
            <p className="mt-0.5 flex flex-wrap gap-x-3 text-[11px]">
              <span className={KIND_STYLE[item.kind] ?? ""}>{item.kind}</span>
              <span className="text-neutral-600">{item.source_type}</span>
              <span className={FRESHNESS_STYLE[item.freshness] ?? ""}>{item.freshness}</span>
              <span className="text-neutral-600">by {item.collected_by}</span>
            </p>
            {item.citations.length ? (
              <p className="mono mt-0.5 text-[11px] text-neutral-500">
                {item.citations
                  .map((c) => c.render ?? citationLabel(c))
                  .filter(Boolean)
                  .join("  ")}
              </p>
            ) : null}
          </div>
        </div>
      </button>

      {open ? (
        <div className="mt-2 space-y-2">
          {item.excerpt ? (
            <pre className="mono max-h-64 overflow-auto rounded bg-neutral-950 p-2 text-neutral-400">
              {item.excerpt}
            </pre>
          ) : null}
          {item.citations.some((c) => c.url) ? (
            <ul className="text-[11px]">
              {item.citations
                .filter((c) => c.url)
                .map((c) => (
                  <li key={c.url}>
                    <a
                      href={c.url!}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-cyan-400 hover:underline"
                    >
                      {c.title || c.url}
                    </a>
                  </li>
                ))}
            </ul>
          ) : null}
          {artifact ? (
            <pre className="mono max-h-80 overflow-auto rounded bg-neutral-950 p-2 text-neutral-500">
              {artifact.slice(0, 20000)}
            </pre>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

function citationLabel(c: { repo?: string | null; path?: string | null; start_line?: number | null; end_line?: number | null; url?: string | null }) {
  if (c.path) {
    const span =
      c.start_line && c.end_line && c.end_line !== c.start_line
        ? `${c.start_line}-${c.end_line}`
        : c.start_line ?? "";
    return `${c.repo ? `${c.repo}:` : ""}${c.path}${span ? `:${span}` : ""}`;
  }
  return c.url ?? "";
}
