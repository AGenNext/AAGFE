# AAGFE Continuous Governance and Kill Switch

AAGFE is the continuous governance enforcement layer for AGenNext, including kill-switch authority.

## Decision

AAGFE continuously evaluates whether the autonomous agent platform should continue, pause, restrict, quarantine, rollback, or stop execution.

It provides governance enforcement across runtime, deployment, agents, tools, payments, handoffs, hooks, context, mind, belief, and platform operations.

## Scope

AAGFE owns:

- continuous governance enforcement
- kill-switch contracts
- emergency stop policy
- pause/resume governance
- platform-wide restriction modes
- degraded-mode governance
- quarantine triggers
- rollback triggers
- unsafe autonomy detection
- governance event records
- escalation contracts

## Boundary

| Component | Responsibility |
|---|---|
| AAGFE | Continuous governance enforcement and kill switch |
| Agent-Platform | Final authority and governance orchestration |
| Agent-Runtime | Must obey AAGFE stop/pause/restrict decisions |
| Agent-deploy | Must obey AAGFE deployment stop decisions |
| Agent-Guard | Supplies adversarial/input risk signals |
| Agent-Monitor | Supplies runtime/security/drift/error signals |
| Agent-Traces | Records governance enforcement events |
| Agent-Policies | Provides operational policies |
| Agent-Constitution | Provides highest-level governance rules |

## Kill switch states

```txt
normal
restricted
paused
quarantined
rollback_required
stopped
```

## Kill switch triggers

```txt
critical_security_gap
prompt_injection_confirmed
malware_detected
unauthorized_deploy_attempt
unauthorized_payment_attempt
context_poisoning_detected
belief_poisoning_detected
runtime_runaway_detected
cost_runaway_detected
compliance_blocker_detected
human_emergency_stop
platform_owner_stop
```

## Rule

AAGFE is not advisory.

Runtime and deployment layers must enforce AAGFE decisions before continuing execution.
