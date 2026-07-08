# AMALI Addendum - Operant / OrderPilot Repo-Agent Discipline

## Purpose

This addendum imports the practical repository-agent discipline previously used in
Operant / OrderPilot-style development into AMALI.

It does not replace the AMALI constitution.

It supplements AMALI with strict rules for:

- repo-aware coding agents;
- DevOps agents;
- GitHub workflow;
- CI/security checks;
- prompt changes;
- test discipline;
- root-cause reports;
- safe PR-based development.

AMALI remains the higher-level architecture:

```text
Adaptive Multi-Agent LLM Infrastructure
with Controlled Self-Improvement,
MoE Routing,
Secure Memory,
Evaluation,
Owner-Governed Deployment,
and Anti-Degradation Control Plane.
```

## 1. Layered Agent Discipline

Every coding/devops/security agent must operate through these layers.

### 1.1 Intent Layer

Before doing anything, identify:

```text
- exact task
- expected output
- affected repository area
- risk level
- whether code, config, docs, CI, or security is involved
- whether the task requires tests
- whether the task can touch GitHub settings
```

If the task is ambiguous, use the safest reasonable interpretation and state
assumptions.

Do not guess silently.

### 1.2 Evidence Layer

The agent must distinguish:

```text
[KNOWN]       Verified by repo files, logs, tests, CI output, official docs, or user-provided evidence.
[INFERRED]    Reasonable engineering inference from evidence.
[UNKNOWN]     Not enough evidence.
[HYPOTHESIS]  Research or design idea requiring validation.
[RISK]        Potential bug, security issue, degradation, or operational hazard.
```

Never present hypotheses as facts.

Never invent:

```text
- test results
- files
- GitHub settings
- benchmark scores
- security guarantees
- model capabilities
- production state
```

### 1.3 Exploration Layer

Before changing code/config, inspect:

```text
1. repository structure
2. current branch
3. git status
4. related files
5. related tests
6. existing naming/style conventions
7. security boundaries
8. CI workflow expectations
9. current failure logs if available
10. rollback path
```

Do not patch before inspecting.

### 1.4 Discipline Layer

The required workflow is:

```text
1. Inspect
2. Diagnose
3. Identify root cause
4. Propose minimal safe patch
5. Apply patch
6. Add/update tests
7. Run targeted checks
8. Run broader checks if needed
9. Report evidence
10. Open PR
```

The agent must prefer:

```text
- minimal patch
- explicit tests
- small commits
- reviewable changes
- no unrelated refactor
- no hidden behavior changes
- no architecture bypass
```

### 1.5 Security Contract Layer

The agent must not:

```text
- commit secrets
- create real .env files
- paste tokens into logs/issues/PRs
- disable security checks to pass CI
- bypass branch protection
- push directly to main
- force push main
- weaken auth/policy boundaries
- add hidden backdoors
- give agents unrestricted shell/network access
- store secrets in memory
- commit models, datasets, logs, caches, embeddings, or vector stores
```

The agent must enforce:

```text
- least privilege
- deny-by-default tooling
- PR-based workflow
- auditability
- rollbackability
- explicit owner approval for high-risk changes
```

## 2. GitHub Workflow Rules

Default local workspace:

```text
D:\AMALI\AMALI-Core
```

Large local artifacts must remain outside git:

```text
D:\AMALI\data
D:\AMALI\models
D:\AMALI\cache
```

The agent must start from repo root:

```powershell
cd D:\AMALI\AMALI-Core
```

Every task must begin with:

```powershell
git status
git branch --show-current
git remote -v
```

Never work from:

```text
D:\AMALI
```

because it may expose non-repo folders such as data, models, cache, or old
workspace folders.

## 3. Branch and PR Rules

Never work directly on `main`.

Required flow:

```powershell
cd D:\AMALI\AMALI-Core
git switch main
git pull origin main
git switch -c <type>/<short-task-name>
```

Examples:

```text
foundation/tooling-ci
docs/prompt-standards
security/github-governance
core/agent-protocol-schemas
```

After changes:

```powershell
git status
git add .
git commit -m "<type>: <clear summary>"
git push -u origin <branch>
gh pr create --title "<title>" --body "<body>"
```

Do not merge automatically unless owner explicitly approves.

## 4. CI / Security Run Discipline

The agent must run or report why it could not run:

```text
pytest
ruff
mypy
bandit
gitleaks
trivy
semgrep
```

For each check, report:

```text
PASS
FAIL
NOT RUN
```

with evidence.

Do not rerun old failed CI after pushing a fix commit. Wait for the new CI run on
the new commit.

Do not delete tests or weaken checks to make CI pass.

## 5. Root Cause Report Standard

Every coding/devops/security task must end with:

```text
## Summary
What changed.

## Evidence
Files, logs, tests, CI, docs used.

## Root Cause
What was actually wrong or missing.

## Patch Summary
What was changed and why.

## Files Changed
List every changed file.

## Tests / Checks
pytest:
ruff:
mypy:
bandit:
gitleaks:
trivy:
semgrep:

## Security Impact
Whether auth/secrets/policy/tooling/CI risk changed.

## Risks
Remaining risks or unknowns.

## Rollback
How to revert safely.

## Next Step
One precise next action.
```

## 6. Prompt Change Rules

Prompts are versioned engineering artifacts.

Any change to prompts, agent instructions, router instructions, safety policies,
or memory policies must include:

```text
- purpose
- affected agent/component
- old behavior
- new behavior
- risk analysis
- eval or regression check
- rollback path
```

Do not silently mutate prompts.

Prompt changes must go through PR.

## 7. AMALI-Specific Additions

Because AMALI is an AI infrastructure, repo-agent discipline must also check:

```text
- evidence labels are preserved
- no fake AGI/ASI claims are introduced
- self-improvement remains controlled and approval-gated
- model/router/prompt/memory/tool policies are versionable
- degradation/fallback behavior is documented
- memory does not override fresh evidence
- no agent gets unrestricted tool access
- no training/fine-tuning starts before eval foundation exists
```

## 8. Output Quality Rules

Agent outputs must be:

```text
- direct
- evidence-grounded
- minimal
- security-aware
- reproducible
- testable
- rollbackable
```

Avoid:

```text
- vague summaries
- fake confidence
- broad rewrites
- hidden assumptions
- unverified claims
- unnecessary complexity
- "trust me" reports
```

## 9. Definition of Done

A task is complete only when:

```text
[KNOWN]
- branch exists
- patch is committed
- PR is opened
- changed files are listed
- tests/checks are run or explicitly marked NOT RUN
- no secrets are committed
- no direct main push happened
- risks are stated
- next step is clear

[NOT ACCEPTABLE]
- untested silent patch
- fake test results
- security check disabled
- unrelated rewrite
- prompt mutation without review
- production/self-improvement behavior added without approval gate
```
