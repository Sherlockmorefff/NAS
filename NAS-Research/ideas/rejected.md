最后更新日期：2026-08-13
当前状态：[已验证] 仅记录有仓库证据的归档/暂停方向；未实现想法不自动视为否决。

# 归档、暂停与否决的想法

## 记录原则

[已验证] 本页用于避免重复走已明确退出正式管线的路径。只有存在实现、审计或明确决策证据时才登记；“没有实现”只表示未知或未来方向，不等同于被否决。

## Geometric acquisition prototype

当前决策：归档，排除在 formal execution gate 之外。

[已验证] [legacy/geometric_acquisition/README.md](../../legacy/geometric_acquisition/README.md) 将 differentiable decoder、Jacobian utilities、`GeometricLogEI` 和 integration experiment 标为 experimental archived branch。[repository_layout.md](../../docs/repository_layout.md) 也明确 current production/formal runner 不执行该目录实现。

归档原因与边界：

- [已验证] 它不是当前 qLogEI 正式路径的依赖，不能混入 S0/G100/G150 方法描述。
- [已验证] 原型代码仍保留，可通过 Git 历史审计；归档不等于删除，也不证明思路在科学上无效。
- [计划] 若重新考虑，必须先定义与标准 qLogEI 的最小对照、数值稳定测试、acquisition overhead 和 formal gate；不得直接复用旧探索结果作为新主张。

## 不在否决清单中的方向

- [未知] Knowledge Gradient 尚未实现，不属于被否决；见 [future.md](future.md)。
- [已实现未充分验证] DKL surrogate 有代码与测试但不是正式方法，也不等同于被否决。
- [计划] 跨数据集 final evaluation 是缺失工作，不是失败方向。

## 新条目的最低信息

[计划] 后续若新增条目，至少记录：日期、决策（暂停/归档/否决）、直接证据路径、可复现失败或资源原因、对正式结果的影响，以及重新开启条件。避免把推测写成结论。
