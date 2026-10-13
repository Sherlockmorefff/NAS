最后更新日期：2026-08-13
当前状态：[已实现未充分验证] TED 引用可核实；qLogEI、Exact GP、BOOM-Explorer 的完整书目信息仍待补齐。

# Bayesian Optimization 文献笔记

## Transductive Experimental Design

[已验证] 仓库协议给出的完整引用为：

Kai Yu, Jinbo Bi, and Volker Tresp. “Active Learning via Transductive Experimental Design.” *Proceedings of the 23rd International Conference on Machine Learning*, 2006. DOI: 10.1145/1143844.1143980.

证据：[wgmm_clustered_ted.md](../../docs/wgmm_clustered_ted.md)。

[已验证] 本项目的 G100/G150 使用 TED-inspired clustered selector。该实现按本项目固定候选域、RBF kernel 与 quota 规则构造，不应称为对论文或外部代码的逐字复现。

## BOOM-Explorer 行为审计

[已验证] 仓库文档记录了 BOOM-Explorer pinned implementation 的行为审计，并明确没有复制其源代码；本项目实现使用不同的普通对角 GMM 与显式核/配额规则。

[未知] 仓库内没有可核实的 BOOM-Explorer 完整作者、年份与正式论文条目，因此此处不补写猜测信息。论文引用前 TODO：从原始论文或项目页面核实完整书目与版本。

## Exact GP 与 qLogEI

[已验证] 正式 S0/G100/G150 实现使用 `SingleTaskGP` 型 scratch Exact GP、fixed-domain normalization 与 qLogEI；这是代码事实，不等同于文献引用。证据：[accuracy_gp.py](../../surrogate/accuracy_gp.py)、[bo_phase4.py](../../bo_phase4.py)、[正式配置](../../configs/cross_dataset_methods.json)。

- [计划] TODO：补齐 Gaussian-process BO、Expected Improvement/log-EI 和 BoTorch 实现依据的原始文献。
- [计划] TODO：区分算法原始论文、数值稳定版本和软件引用，避免把软件 API 当作理论来源。

## Knowledge Gradient

[未知] 仓库无 KG 实现、配置、结果或可核实文献资产。当前只存在 [对照实验设计](../experiments/KG.md)，不得列作已完成 baseline。

## DKL

[已实现未充分验证] 仓库有 DKL surrogate 代码与测试，但它不在正式 S0/G100/G150 配置中。TODO：若论文讨论 DKL，先补原始引用并明确它是诊断/非正式实验，而非主结果方法。
