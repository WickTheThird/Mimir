// Mirrors the pydantic models in src/mimir/models. Kept narrow on purpose: the
// UI reads what it renders and ignores the rest, so a backend field addition
// does not break the build.

export type RiskClass = "R0" | "R1" | "R2" | "R3" | "R4";

export interface Citation {
  render?: string;
  path?: string | null;
  repo?: string | null;
  start_line?: number | null;
  end_line?: number | null;
  url?: string | null;
  title?: string | null;
}

export interface Evidence {
  id: string;
  claim: string;
  kind: "observed" | "inferred" | "hypothesis";
  source_type: string;
  source_id: string;
  excerpt: string;
  citations: Citation[];
  freshness: string;
  confidence: number;
  supports: boolean;
  collected_by: string;
  artifact_ref?: string | null;
}

export interface Hypothesis {
  id: string;
  statement: string;
  status: string;
  likelihood: number;
  next_check?: string | null;
  rejected_reason?: string | null;
}

export interface FinalAnswer {
  answer: string;
  confidence: number;
  observed_facts: string[];
  inferences: string[];
  unverified: string[];
  disagreements: string[];
  citations: string[];
  next_steps: string[];
  proposed_commands: string[];
}

export interface PendingApproval {
  id: string;
  session_id: string | null;
  command: string;
  risk: RiskClass;
  production_target: boolean;
  reversible: boolean;
  rollback_hint: string | null;
  reasons: string[];
  context: Record<string, unknown>;
  purpose: string;
  expected_effect: string;
  prompt: string;
  created_at: number;
  expires_at: number | null;
}

export type EventType =
  | "started"
  | "node_end"
  | "plan"
  | "specialist"
  | "evidence"
  | "command"
  | "approval"
  | "answer"
  | "error"
  | "done";

export interface RunEvent {
  type: EventType;
  at: number;
  [key: string]: unknown;
}

export interface SpecialistEvent {
  specialist: string;
  conclusion: string;
  confidence: number;
  tool_calls: number;
  evidence: number;
  error: string | null;
}

export interface PlanEvent {
  task_type: string;
  steps: { specialist: string; objective: string }[];
  skills: string[];
  missing_context: string[];
}

export interface CommandEvent {
  id: string;
  display: string;
  risk: RiskClass | null;
  preview: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  status: string;
  created_at: number;
  user_request: string;
  interface: string;
}
