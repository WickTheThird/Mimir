import type { ReactNode } from "react";
import type { RiskClass } from "../types";

const RISK_STYLE: Record<string, string> = {
  R0: "bg-neutral-800 text-neutral-400 border-neutral-700",
  R1: "bg-emerald-950 text-emerald-400 border-emerald-800",
  R2: "bg-amber-950 text-amber-400 border-amber-800",
  R3: "bg-orange-950 text-orange-400 border-orange-800",
  R4: "bg-red-950 text-red-300 border-red-700",
};

const RISK_LABEL: Record<string, string> = {
  R0: "analysis only",
  R1: "read-only",
  R2: "elevated inspection",
  R3: "reversible mutation",
  R4: "high risk",
};

export function RiskBadge({ risk }: { risk: RiskClass | null | undefined }) {
  if (!risk) return null;
  return (
    <span
      title={RISK_LABEL[risk]}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium ${
        RISK_STYLE[risk] ?? RISK_STYLE.R0
      }`}
    >
      {risk}
      <span className="opacity-70">{RISK_LABEL[risk]}</span>
    </span>
  );
}

export function Confidence({ value }: { value: number }) {
  const label =
    value >= 0.75
      ? "well supported"
      : value >= 0.5
        ? "verify key claims"
        : value >= 0.3
          ? "a lead, not a conclusion"
          : "gather more evidence";
  const tone =
    value >= 0.75
      ? "text-emerald-400"
      : value >= 0.5
        ? "text-amber-400"
        : value >= 0.3
          ? "text-orange-400"
          : "text-red-400";
  return (
    <span className={`text-xs ${tone}`}>
      confidence {value.toFixed(2)} <span className="opacity-70">({label})</span>
    </span>
  );
}

export function Panel({
  title,
  children,
  right,
  className = "",
}: {
  title?: ReactNode;
  children: ReactNode;
  right?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {title ? (
        <header className="flex items-center justify-between border-b border-neutral-800 px-3 py-2">
          <h2 className="text-xs uppercase tracking-wide text-neutral-500">{title}</h2>
          {right}
        </header>
      ) : null}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-neutral-600">{children}</p>;
}
