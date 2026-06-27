# Graph Report - .  (2026-06-22)

## Corpus Check
- Corpus is ~1,072 words - fits in a single context window. You may not need a graph.

## Summary
- 26 nodes · 24 edges · 6 communities (4 shown, 2 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_AMALI Architecture|AMALI Architecture]]
- [[_COMMUNITY_PR Security Workflow|PR Security Workflow]]
- [[_COMMUNITY_Agent Discipline Layers|Agent Discipline Layers]]
- [[_COMMUNITY_AMALI Safety Rules|AMALI Safety Rules]]
- [[_COMMUNITY_Addendum Scope|Addendum Scope]]
- [[_COMMUNITY_CI Definition Done|CI Definition Done]]

## God Nodes (most connected - your core abstractions)
1. `AMALI` - 7 edges
2. `Layered Agent Discipline` - 5 edges
3. `Foundation Before Training` - 3 edges
4. `Security Contract Layer` - 3 edges
5. `AMALI-Specific Additions` - 3 edges
6. `PR-Based Workflow` - 3 edges
7. `Evaluation` - 2 edges
8. `AMALI Addendum` - 2 edges
9. `Branch and PR Rules` - 2 edges
10. `Eval Foundation Before Training` - 2 edges

## Surprising Connections (you probably didn't know these)
- `Eval Foundation Before Training` --semantically_similar_to--> `Foundation Before Training`  [INFERRED] [semantically similar]
  AGENTS.md → README.md
- `AMALI Addendum` --references--> `AMALI`  [EXTRACTED]
  AGENTS.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **AMALI Architecture Components** — readme_amali, readme_controlled_self_improvement, readme_moe_routing, readme_secure_memory, readme_evaluation, readme_owner_governed_deployment [EXTRACTED 1.00]
- **Repo Agent Operating Discipline** — agents_intent_layer, agents_evidence_layer, agents_exploration_layer, agents_discipline_layer, agents_security_contract_layer [EXTRACTED 1.00]
- **Safe PR-Based Development** — agents_github_workflow_rules, agents_branch_and_pr_rules, agents_ci_security_run_discipline, agents_prompt_change_rules, agents_definition_of_done [EXTRACTED 1.00]

## Communities (6 total, 2 thin omitted)

### Community 0 - "AMALI Architecture"
Cohesion: 0.33
Nodes (7): AMALI, Controlled Self-Improvement, Evaluation, Foundation Before Training, MoE Routing, Owner-Governed Deployment, Secure Memory

### Community 1 - "PR Security Workflow"
Cohesion: 0.33
Nodes (6): Branch and PR Rules, GitHub Workflow Rules, Least Privilege, PR-Based Workflow, Prompt Change Rules, Security Contract Layer

### Community 2 - "Agent Discipline Layers"
Cohesion: 0.40
Nodes (5): Discipline Layer, Evidence Layer, Exploration Layer, Intent Layer, Layered Agent Discipline

### Community 3 - "AMALI Safety Rules"
Cohesion: 0.50
Nodes (4): AMALI-Specific Additions, Approval-Gated Self-Improvement, Eval Foundation Before Training, Evidence Labels

## Knowledge Gaps
- **8 isolated node(s):** `Controlled Self-Improvement`, `MoE Routing`, `Secure Memory`, `Owner-Governed Deployment`, `Repo-Agent Discipline` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AMALI` connect `AMALI Architecture` to `Addendum Scope`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Why does `Foundation Before Training` connect `AMALI Architecture` to `AMALI Safety Rules`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `Layered Agent Discipline` connect `Agent Discipline Layers` to `PR Security Workflow`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **What connects `Controlled Self-Improvement`, `MoE Routing`, `Secure Memory` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._