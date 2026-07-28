import type { FinalAnswer, Hypothesis } from "../types";
import { Confidence, Empty, Panel } from "./primitives";

/**
 * Evidence-first answer layout (ADR 2, 21.3): observed, inferred, and
 * unverified are separated structurally rather than blended into prose.
 */
export function AnswerPanel({
  answer,
  confidence,
  hypotheses,
}: {
  answer: FinalAnswer | null;
  confidence: number;
  hypotheses: Hypothesis[];
}) {
  if (!answer) return <Empty>No answer yet.</Empty>;

  return (
    <div className="space-y-4">
      <div>
        <p className="whitespace-pre-wrap text-sm text-neutral-100">{answer.answer}</p>
        <div className="mt-2">
          <Confidence value={confidence} />
        </div>
      </div>

      <Section title="observed" tone="text-emerald-400" items={answer.observed_facts} />
      <Section title="inferred" tone="text-amber-400" items={answer.inferences} />
      <Section title="unverified" tone="text-orange-400" items={answer.unverified} />
      <Section
        title="disagreement between specialists"
        tone="text-red-400"
        items={answer.disagreements}
      />

      {answer.proposed_commands.length ? (
        <div>
          <h3 className="mb-1 text-xs uppercase tracking-wide text-neutral-500">
            proposed commands
          </h3>
          {answer.proposed_commands.map((command) => (
            <pre
              key={command}
              className="mono mb-1 overflow-x-auto rounded bg-neutral-950 p-2 text-neutral-200"
            >
              $ {command}
            </pre>
          ))}
        </div>
      ) : null}

      {hypotheses.length ? (
        <Panel title="hypotheses">
          <ul className="space-y-1">
            {hypotheses.map((h) => (
              <li key={h.id} className="text-xs">
                <span className="text-neutral-500">{h.likelihood.toFixed(2)}</span>{" "}
                <span
                  className={
                    h.status === "rejected"
                      ? "text-neutral-600 line-through"
                      : "text-neutral-200"
                  }
                >
                  {h.statement}
                </span>
                {h.next_check ? (
                  <span className="ml-2 text-neutral-500">next: {h.next_check}</span>
                ) : null}
                {h.rejected_reason ? (
                  <span className="ml-2 text-neutral-600">({h.rejected_reason})</span>
                ) : null}
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}

      <Section title="next steps" tone="text-cyan-400" items={answer.next_steps} />
      <Section title="citations" tone="text-neutral-500" items={answer.citations} mono />
    </div>
  );
}

function Section({
  title,
  items,
  tone,
  mono = false,
}: {
  title: string;
  items: string[];
  tone: string;
  mono?: boolean;
}) {
  if (!items?.length) return null;
  return (
    <div>
      <h3 className="mb-1 text-xs uppercase tracking-wide text-neutral-500">{title}</h3>
      <ul className="space-y-0.5">
        {items.map((item, index) => (
          <li key={index} className={`text-xs ${tone} ${mono ? "mono" : ""}`}>
            - {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
