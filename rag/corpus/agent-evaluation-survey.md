# Research Note: Survey on Evaluation of LLM-based Agents

Source: arXiv:2503.16416, **Survey on Evaluation of LLM-based Agents**

Authors: Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, Michal Shmueli-Scheuer

Initial submission: March 20, 2025  
Latest version referenced: v2, April 23, 2026  
Venue note: ACL Findings

## Why this matters for AGenNext

The paper surveys evaluation methods for LLM-based agents, which are autonomous systems that can plan, reason, use tools, and interact with dynamic environments.

It analyzes agent evaluation across five perspectives:

1. Core LLM capabilities required for agentic workflows, such as planning and tool use.
2. Application-specific benchmarks, including web and software-engineering agents.
3. Evaluation of generalist agents.
4. Core dimensions of agent benchmarks.
5. Evaluation frameworks and tools for agent developers.

The paper highlights gaps in cost-efficiency, safety, robustness, and fine-grained scalable evaluation methods.

## Product implications

AGenNext should not only govern agent access. It should also capture evaluation evidence that helps determine whether an agent is safe, reliable, and trustworthy enough to receive autonomy.

This directly supports the following platform modules:

- **Agent Trust** — agent evaluation evidence becomes a trust signal.
- **Agent Risk Score** — weak safety, robustness, or benchmark performance should increase risk.
- **Agent Trace** — runtime traces become evaluation evidence after deployment.
- **Agent Security** — safety and robustness gaps become threat signals.
- **Agent Access Review** — reviewers should see agent evaluation posture before approving access.

## Suggested evaluation dimensions for AGenNext

| Dimension | Governance use |
|---|---|
| Planning quality | Determines allowed autonomy level |
| Tool-use reliability | Influences access to sensitive tools |
| Robustness | Impacts production-readiness and risk score |
| Safety behavior | Drives policy restrictions and approval requirements |
| Cost-efficiency | Helps detect wasteful or runaway agents |
| Benchmark freshness | Prevents stale evaluation claims |
| Trace consistency | Confirms whether runtime behavior matches evaluation |
| Application-specific performance | Controls access to domain-specific workflows |

## Proposed product feature: Agent Evaluation Evidence

AGenNext should introduce an `AgentEvaluationEvidence` object that records pre-deployment and runtime evaluation data.

```json
{
  "agent_id": "agent:sales-copilot-17",
  "evaluation_id": "eval_2026_04_sales_support",
  "benchmark_type": "application_specific",
  "capabilities": {
    "planning": "pass",
    "tool_use": "pass",
    "robustness": "warning",
    "safety": "pass",
    "cost_efficiency": "warning"
  },
  "governance_effect": {
    "trust_score_delta": -4,
    "risk_score_delta": 8,
    "required_controls": [
      "trace_required",
      "human_approval_for_bulk_actions"
    ]
  }
}
```

## Architecture connection

```text
Agent Evaluation Evidence
   ↓
Agent Trust
   ↓
Agent Risk Score
   ↓
Runtime Authorization
   ↓
Agent Trace
   ↓
Continuous Evaluation Feedback
```

## README positioning

AGenNext should describe evaluation as a governance input:

> AGenNext converts agent evaluation results, runtime traces, safety signals, and access behavior into trust, risk, and authorization controls.

