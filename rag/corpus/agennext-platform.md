# AGenNext Agent Governance Platform

AGenNext is a standards-aligned governance control plane for autonomous AI agents.

It helps enterprises understand what agents can access, what they actually did, how risky they are, what secrets they can use, and which controls should apply before agent actions become business impact.

## Core product pillars

- **Agent Bridge** — connects agent runtimes, identity providers, policy engines, secret managers, cloud apps, and observability tools.
- **Agent Fabric** — maps the graph of agents, users, groups, tools, apps, resources, secrets, policies, paths, threats, and traces.
- **Agent Trust** — evaluates whether an agent is known, owned, reviewed, signed, governed, and reliable enough to act.
- **Agent Vault** — brokers access to credentials, secrets, tokens, keys, sensitive memory, and protected artifacts.
- **Agent Security** — detects and responds to threats across tools, prompts, access paths, runtime behavior, and cloud actions.
- **Agent Risk Score** — continuously scores each agent based on privilege, autonomy, data exposure, behavior, and trace coverage.
- **Agent Access Review** — certifies, constrains, or revokes agent access across tools, data, SaaS apps, and delegated authority.
- **Agent Trace** — records tamper-evident evidence for actions, authorization decisions, secret access, approvals, and policy outcomes.

## Standards-aligned architecture

AGenNext is designed around open and pluggable governance primitives:

- AuthZEN-style authorization decision APIs
- Open Policy Agent / Rego policy backends
- OpenFGA-style relationship authorization
- Secret-manager integrations such as Bitwarden, Vault, and cloud KMS
- OpenTelemetry/SIEM-friendly trace export

## Local preview

Open `index.html` in a browser, or serve the folder with any static server:

```bash
python3 -m http.server 8080
```

Then visit `http://localhost:8080`.

## Suggested next steps

- Replace mock metrics with live API data.
- Add AuthZEN decision-service endpoint.
- Add OPA and OpenFGA connectors.
- Add SaaS/cloud app discovery connectors.
- Add authenticated dashboard routes.
- Deploy with GitHub Pages, Vercel, Netlify, or Cloudflare Pages.
