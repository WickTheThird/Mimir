// API client. Every call goes to the local MIMIR service; the UI holds no
// credentials of its own and never reaches a privileged tool directly (ADR 15).

import type { PendingApproval, RunEvent, SessionSummary } from "./types";

const BASE = "/api";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail.slice(0, 400)}`);
  }
  return (await response.json()) as T;
}

export interface InvestigateInput {
  question: string;
  cluster_context?: string;
  namespace?: string;
  repositories?: string[];
  sdm_resource?: string;
  time_range?: string;
}

/**
 * Stream an investigation. Returns an abort handle so navigating away or
 * starting a new run cancels the in-flight one rather than leaking it.
 */
export function streamInvestigation(
  input: InvestigateInput,
  onEvent: (event: RunEvent) => void,
  onError: (message: string) => void,
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${BASE}/investigations/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        onError(`${response.status} ${response.statusText}`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // Normalise line endings first. The SSE spec allows CRLF, LF, or CR,
        // and sse-starlette emits CRLF, so a parser that only looks for "\n\n"
        // silently consumes the whole stream and yields nothing.
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");

        // SSE frames are separated by a blank line. A frame can span reads, so
        // only complete frames are consumed and the remainder stays buffered.
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const dataLines = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim());
          if (dataLines.length) {
            try {
              onEvent(JSON.parse(dataLines.join("\n")) as RunEvent);
            } catch {
              // A ping or malformed frame; ignore rather than killing the stream.
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        onError((error as Error).message);
      }
    }
  })();

  return () => controller.abort();
}

export const api = {
  capabilities: () => json<{ tools: unknown[]; safety: Record<string, unknown> }>("/capabilities"),
  sessions: () => json<{ sessions: SessionSummary[] }>("/sessions"),
  session: (id: string) => json<Record<string, unknown>>(`/sessions/${id}`),
  exportSession: (id: string, fmt = "md") =>
    json<{ content: string }>(`/sessions/${id}/export?fmt=${fmt}`),
  approvals: () => json<{ approvals: PendingApproval[] }>("/approvals"),
  decide: (id: string, decision: string, extra: Record<string, unknown> = {}) =>
    json<Record<string, unknown>>(`/approvals/${id}`, {
      method: "POST",
      body: JSON.stringify({ decision, ...extra }),
    }),
  artifact: (ref: string) => json<{ content: string }>(`/artifacts/${ref}`),
  skills: () => json<{ skills: { name: string; when_to_use: string }[] }>("/skills"),
  searchMemory: (q: string) =>
    json<{ results: Record<string, unknown>[] }>(`/memory/search?q=${encodeURIComponent(q)}`),
};
