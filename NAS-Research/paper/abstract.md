---
last_updated: 2026-08-14
status: living-draft
claims: [C001, C002, C003, C006, C009, C010, C011, C012]
evidence: [E001, E003]
literature: [D-VAE, TED]
---

# Living abstract

## 背景与问题

Joint neural architecture search and hyperparameter optimization for graph neural networks is expensive because every observation requires training a concrete architecture–hyperparameter configuration. We study whether the initialization of Bayesian optimization changes coverage, surrogate learning, cost-aware sample efficiency, and final performance in this joint domain.

## 方法

Our current system searches a 12-dimensional architecture D-VAE latent together with four direct normalized hyperparameter coordinates. It compares a global Schur baseline with clustered TED/Schur and low-fidelity promotion strategies, followed by the same scratch Exact GP and qLogEI online search. All methods use 300 full evaluations, while low-fidelity epochs are reported as additional cost.

## 实验范围

[已验证] Current artifacts cover Cora S0/G100/G150 searches and a shared validation-selected final-evaluation protocol, a separate Cora K6-LF protocol, and historical S0/G100/G150 searches on CiteSeer, PubMed, DBLP, and Flickr. [WAITING-FOR-E004] Matching cross-dataset final evaluations are unavailable. [WAITING-FOR-E005] Knowledge Gradient has not been implemented or run.

## 暂定发现

[已验证] On Cora, the three main methods have similar 300-full search-validation means and overlapping uncertainty intervals; validation-selected final-test results do not support a stable improvement from G100/G150 over S0. [已验证] In the historical four-dataset search matrix, S0 has the highest mean search validation on all four datasets, while surrogate diagnostics show no method that consistently dominates across metrics and datasets. [实验进行中] Whether structured initialization improves early sample efficiency after accounting for low-fidelity and wall-clock cost remains [TBD-RESULT].

## 潜在意义

[假设] These results may support an evidence-tiered evaluation of BO initialization rather than a universal performance claim: full-budget equality does not imply equal computation, surrogate improvements need not transfer to final test, and cross-dataset search validation cannot substitute for final evaluation.

当前可写：已核实的实验范围、Cora negative/boundary result、四数据集 search-only 边界。当前禁止：显著提升、统计等价、跨数据集 test 泛化、KG 收益、coverage 或 early-efficiency 改善。
