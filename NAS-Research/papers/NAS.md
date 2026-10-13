最后更新日期：2026-08-13
当前状态：[已实现未充分验证] 仅收录仓库内可核实的 D-VAE 文献；NAS 相关工作覆盖仍是骨架。

# NAS 文献笔记

## D-VAE

[已验证] 仓库 README 给出的引用为：

Muhan Zhang, Shali Jiang, Zhicheng Cui, Roman Garnett, and Yixin Chen. “D-VAE: A Variational Autoencoder for Directed Acyclic Graphs.” *Advances in Neural Information Processing Systems 32*, 2019, pp. 1586–1598. arXiv:1904.11088.

仓库证据：[README.md](../../README.md)。

[已验证] 本项目使用 D-VAE 风格的 DAG architecture encoder/decoder；当前架构 latent 为 12 维。论文中应将该引用用于 architecture representation，而不能据此把 Phase4 的完整 16D 搜索坐标都称作 D-VAE latent。实现边界见 [项目总览](../project_overview.md#联合搜索空间与术语边界)。

## 与本文的关系

[已验证] D-VAE 提供结构化架构表示；本项目在该 12D 表示旁拼接 4D direct HP coordinates，并在联合域上研究初始化和 BO。项目贡献不应表述为提出 D-VAE 本身。

## 待补文献

- [计划] TODO：补齐 graph NAS、joint NAS+HPO、latent-space NAS 与 predictor/surrogate-based NAS 的原始论文。
- [计划] TODO：为每篇文献记录研究问题、搜索空间、surrogate/acquisition、预算、数据集、统计单位和与本项目差异。
- [计划] TODO：引入正式 `.bib` 前逐条核对 DOI/arXiv/venue；本轮不联网补全。

[未知] 当前仓库没有独立论文 PDF、`.bib` 或系统阅读笔记，因此不能声称相关工作综述已完备。
