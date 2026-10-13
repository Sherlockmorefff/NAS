---
last_updated: 2026-08-14
status: waiting-manifest-extraction
claims: [C010]
evidence: [E001, E003]
literature: []
---

# T06 — Dataset statistics candidate

- 研究问题：Experimental setup / RQ4 boundary。
- 数据来源：每个正式 artifact 的 `dataset_manifest.json`；不得凭记忆或从论文网页手填。
- 统计单位：运行时实际 graph/dataset identity。
- 当前完整度：[计划] manifest paths 已登记，统计值未写入项目级事实源。
- 进入论文：否，等待核实与数据集引用。

| Dataset | Nodes | Edges | Features | Classes | Train/Val/Test | Split protocol | Transform | Manifest | Citation status |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| Cora | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | Planetoid public | NormalizeFeatures | TODO exact artifact | TODO |
| CiteSeer | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | Planetoid public | NormalizeFeatures | E003 leaf manifests | TODO |
| PubMed | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | Planetoid public | NormalizeFeatures | E003 leaf manifests | TODO |
| DBLP | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | 20/class, 500, 1000 | deterministic seed-0 constructed | NormalizeFeatures | E003 leaf manifests | TODO |
| Flickr | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | official masks | official | none | E003 leaf manifests | TODO |

- 当前能支持：loader/split/transform 类型。
- 当前不能支持：任何未抽取统计值或完整 dataset bibliography。
