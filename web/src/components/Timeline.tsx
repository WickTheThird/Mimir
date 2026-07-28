import type { CommandEvent, PlanEvent, RunEvent, SpecialistEvent } from "../types";
import { RiskBadge, Empty } from "./primitives";

/** Specialist activity timeline (ADR 15). Shows what ran, in order, honestly. */
export function Timeline({ events }: { events: RunEvent[] }) {
  const interesting = events.filter((e) =>
    ["plan", "specialist", "command", "error", "started", "done"].includes(e.type),
  );
  if (!interesting.length) return <Empty>Nothing has run yet.</Empty>;

  return (
    <ol className="space-y-2">
      {interesting.map((event, index) => (
        <li key={index} className="border-l-2 border-neutral-800 pl-3">
          <Entry event={event} />
        </li>
      ))}
    </ol>
  );
}

function Entry({ event }: { event: RunEvent }) {
  switch (event.type) {
    case "started":
      return (
        <p className="text-xs text-neutral-500">
          investigation started
          <span className="mono ml-2 text-neutral-600">{String(event.session_id)}</span>
        </p>
      );
    case "plan": {
      const plan = event as unknown as PlanEvent;
      return (
        <div>
          <p className="text-xs text-neutral-400">
            plan <span className="font-medium text-neutral-200">{plan.task_type}</span>
          </p>
          <ul className="mt-1 space-y-0.5">
            {plan.steps?.map((step, i) => (
              <li key={i} className="text-xs text-neutral-500">
                <span className="text-cyan-400">{step.specialist}</span> {step.objective}
              </li>
            ))}
          </ul>
          {plan.missing_context?.length ? (
            <ul className="mt-1 space-y-0.5">
              {plan.missing_context.map((q) => (
                <li key={q} className="text-xs text-amber-400">
                  needs: {q}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      );
    }
    case "specialist": {
      const report = event as unknown as SpecialistEvent;
      return (
        <div>
          <p className="text-xs">
            <span className={report.error ? "text-red-400" : "text-cyan-400"}>
              {report.specialist}
            </span>
            <span className="ml-2 text-neutral-500">
              conf {report.confidence.toFixed(2)} - {report.tool_calls} tool calls -{" "}
              {report.evidence} evidence
            </span>
          </p>
          {report.conclusion ? (
            <p className="mt-0.5 text-xs text-neutral-400">{report.conclusion}</p>
          ) : null}
          {report.error ? (
            <p className="mt-0.5 text-xs text-red-400">failed: {report.error}</p>
          ) : null}
        </div>
      );
    }
    case "command": {
      const command = event as unknown as CommandEvent;
      return (
        <div className="flex items-start gap-2">
          <RiskBadge risk={command.risk} />
          <code className="mono text-neutral-300">{command.display}</code>
        </div>
      );
    }
    case "error":
      return <p className="text-xs text-red-400">error: {String(event.error)}</p>;
    case "done":
      return (
        <p className="text-xs text-neutral-500">
          done in {String(event.duration_s)}s, {String(event.evidence)} evidence items
        </p>
      );
    default:
      return null;
  }
}
