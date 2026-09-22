# The deterministic gates on the final answer

Four cases failed in every run of every model measured on this corpus, from
7B to 117B. A seventeen-fold parameter increase moved none of them, which
rules out the usual explanation. They were not the model failing to be
clever enough.

Each one turned out to be a fact the system already had and did not carry
into the answer. What follows is what each gate computes, and why it is a
gate rather than a better prompt.

A prompt asking the model to be careful about absence is still the model
deciding, and the measurement says the model decides this wrongly at every
size. A gate decides it the same way every run.

## Sufficiency: you cannot prove absence from a search that did not run

`verify/sufficiency.py`

Asked whether a billing pod exists, MIMIR answered "no" both when a complete
search found nothing and when all three clusters timed out and produced no
listing. Those are opposite situations. The first is a finding. The second
is a failure to look, and on call it means we are blind rather than clear.

The retrieval is classified as `observed`, `empty`, `failed` or `unknown`,
and definite existence claims resting on the last two are demoted to
unverified with the answer prose rewritten to say unknown.

Failure beats completion when both appear. A search where two clusters
answered and the third timed out cannot establish that the third holds
nothing, and reading a partial result as complete is the error itself.

The gate fires only on an explicit failure to look. Silence about whether a
search ran does not count, and that restriction was not the first design.
The first version also demoted on `unknown`, which sounds careful and is
not: forty-seven of the fifty-two model cases in this corpus classify as
unknown, because a prompt describing a situation rarely narrates whether a
search ran. Since "the pod is running" is a presence claim and most ops
answers contain one, the gate would have fired on nearly every case. A gate
that fires on everything is not a gate. It was caught by counting before the
sweep rather than by reading the sweep afterwards.

**Why the claim gate could not catch it.** `verify/claims.py` asks whether
each stated fact resolves to evidence. "There is no billing pod" is a claim
about the *absence* of evidence, so it resolves to nothing by construction
and passes cleanly. That is how the failure survived every model size.

## Grounding: an empty listing names nothing

`verify/grounding.py`

Given a listing that returned no pods at all, MIMIR named pods.

The check for this already existed, already worked, and was already called
by the agent loop. The investigation graph never called it, and the
investigation graph is the path the ops corpus runs. Wiring it into
`synthesise` was the entire fix.

What the operator asked is ground truth alongside what was read. Repeating
back a workload name they supplied is not an invention, and flagging it
would teach them to ignore the warning.

## Currency: a note nobody verified in fourteen months is a lead

`verify/sufficiency.py`

A stored note written fourteen months ago and never verified was treated
exactly like one verified three days ago. The timestamp was already stored
and `compute_freshness` already turned it into a verdict. Neither reached
the answer.

The remembered value is still reported, because it is the most useful thing
available. What changes is its status: a lead to confirm rather than a
reading to act on. An answer that already tells the operator to verify is
left alone, since appending a second instruction to an answer that gave the
right one reads as a system that does not understand its own output.

## Retry signature: one request is not a storm

`verify/patterns.py`

Shown a single request followed by pool exhaustion, with a batch job that
opened four hundred connections in plain view, MIMIR diagnosed a retry
storm. Shown a real storm it said the same thing, so the diagnosis carried
no information either way.

Retries with backoff leave a signature: the same identifier several times,
with the gap between attempts roughly doubling. Read from real timestamps
in production, from the stated situation otherwise.

Three attempts minimum. Two points make one gap, and a single gap cannot
fail to be consistent with doubling, so a two-attempt threshold would match
any pair of lines in any log. The doubling tolerance is wide, 1.4x to 4x,
because real backoff carries jitter and a loaded scheduler stretches gaps
further. This gate catches the answer with no pattern at all; it does not
grade a pattern's tidiness.

**One-directional.** Finding the signature is consistent with retries having
caused the incident but does not establish it, so a match never raises
confidence. Only its absence demotes.

## Rules these four share

**Demotion, never deletion.** The claim may well be true. What is false is
presenting it as established. Relabelling keeps it in front of the operator
while making its status honest. A gate that improves its score by producing
emptier answers has optimised the metric rather than the system.

**The prose is rewritten, not only the bullets.** A demoted bullet under an
answer that still reads "there is no billing pod" leaves the wrong
conclusion in the line that actually gets read.

**Confidence is capped, not scaled.** An answer that invents a workload name
is not slightly less reliable, it is a different kind of thing, and a number
that still reads as fairly confident invites action.

**No gate consults a model.** That is the point of the exercise. The
components of MIMIR that are already deterministic are the only ones that do
not change their mind between identical runs, and per-case churn on this
corpus is roughly a fifth.

## Two defects found while building them

Both were the shape this project keeps finding, where the working path and
the broken path produce identical output.

The freshness window was read from `settings.memory.stale_after_days`, which
does not exist. `getattr` returned the literal written beside it, so the gate
ran on a 30-day window instead of the configured 180 with nothing logged.

The retry gap parser was non-greedy and read "gaps of 1s, 2s and 4s" as two
gaps. The verdict came out right anyway, which is how a parsing bug survives
a passing test. The test now asserts on the parsed list, not only on the
conclusion drawn from it.

## Measurement

Two runs on qwen3-coder:30b, the six absence, ground and fresh cases.

**First, gates alone (21 September):** 3 of 6, and one healthy twin
regressed. The gates fired; the causes were a regex that read a runbook as
an incident and a demotion that kept the false sentence in the prose.

**Second, gates plus the decision layer (22 September):** 6 of 6. The
three cases that had failed on every model from 7B to 117B pass, and their
healthy twins still pass. Three ordering fixes stood between the runs, and
each was found from the stored session metadata, not from the score: the
operator's own explicit statement outranks the decider; the statement about
a record outranks unrelated structured freshness; "cannot be verified"
describes the problem and is not the instruction.

Six cases and one run per model: a direction. The 82-case sweep with every
step live is the number, and it is compared by failure category against
the three prior runs so that a shifted category is distinguishable from
churn.
