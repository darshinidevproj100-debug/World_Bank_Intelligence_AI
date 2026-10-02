# Decision Layer (Provisional)

## Formal definition audit

The repository, README, traceability file, configuration, code, and notebooks contain no authoritative expansion, formula, thresholds, or reference specification for “JEV”. The project request explicitly forbids inventing one. This implementation is a **provisional decision-policy prototype**, not a validated JEV algorithm.

The project owner must provide the official meaning, feature definitions, score/rule formula, thresholds, risk mapping, expected action set, and reference test cases before this can be claimed as formal JEV implementation.

## Interface and behavior

`DecisionInput` carries the question, detected intent, candidates, evidence and coverage, confidence/source, risk, task state, missing details, budgets, previous actions, errors, permissions, ambiguity, and contradiction flag. `DecisionOutput` carries an enumerated action, selected permitted agent, reason, evidence requirement, confidence, risk, stopping flag, next step, and metadata.

The deterministic `PrototypeDecisionPolicy` can request clarification, human review for configured high risk or conflicting evidence, bounded retry/retrieval, agent routing, partial return, or stop. It checks remaining steps and agent-call limits before work, and rejects disallowed candidate agents. A retry is emitted once per retry chain. Confidence is marked measured only for explicit measured/calibrated/human-verified sources; this workflow does not estimate model confidence.

Rules and the evidence threshold are constructor configuration. The workflow currently estimates evidence coverage only from source presence; that is a traceable heuristic, not a semantic confidence score. A stable policy agreement score in evaluation tests agreement with labels only; it cannot validate a formal JEV model.

## Trace records

Each workflow run includes execution ID, stage timestamps, decision ID, policy mode, candidate agents, evidence references/count, decision reason/action, confidence source, risk, remaining budget, decision duration, validation, status, and end-to-end latency. JSONL writes are local and optional failure is surfaced in workflow errors. Do not put credentials or unnecessary personal data in questions or evidence metadata.
