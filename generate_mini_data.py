"""
generate_mini_data.py  v5
=========================
Search Space v3: max_n=7 (5 算子节点), nvt=8 (含 GCNII=type 7)
HP 向量升级为条件参数：[log_lr_norm, dropout_norm, gat_heads_norm, sage_aggr_norm]

v5 变更（相对 v4）：
  - HP 从普通 4 维改为条件参数 4 维，与 nas_space.py v5 对齐：
      [0] log_lr_norm     ∈ [0, 1]，映射至 log_lr ∈ [-4.0, -1.5]
      [1] dropout_norm    ∈ [0, 1]，映射至 dropout ∈ [0.1, 0.6]
      [2] gat_heads_norm  ∈ [0, 1]，仅当架构包含 GATConv(type=3) 时生效
      [3] sage_aggr_norm  ∈ [0, 1]，仅当架构包含 SAGEConv(type=4) 时生效
  - 对无 GAT/SAGE 的样本，将对应条件位置 0，避免无效条件参数随机漂移。
  - 输出文件默认命名为 mini_gnn_dataset_v5.pkl
  - 向后兼容：可通过 --hp_dim=2 生成旧格式（仅 log_lr + dropout）

节点编号 (7 节点, 索引 0~6)：
  0 = START
  1 = Node1 (算子, 禁止 Identity)
  2 = Node2 (算子/Identity)
  3 = Node3 (算子/Identity)
  4 = Node4 (算子/Identity)
  5 = Node5 (算子, 禁止 Identity)
  6 = END
"""

import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

# ============================================================
# 节点类型常量（与 nas_space.py 保持一致）
# ============================================================
START_TYPE    = 0
END_TYPE      = 1
# v3: 新增 GCNII = 7
REAL_OP_TYPES = [2, 3, 4, 5, 7]    # GCNConv, GATConv, SAGEConv, GINConv, GCNII
ALL_OP_TYPES  = [2, 3, 4, 5, 6, 7] # 上述 + Identity

# ============================================================
# HP 采样范围（与 BO 脚本默认值保持一致）
# ============================================================
LOG_LR_MIN    = -4.0
LOG_LR_MAX    = -1.5
DROPOUT_MIN   = 0.1
DROPOUT_MAX   = 0.6


def _sample_hp4(types: list) -> list:
    """采样 4 维条件参数：[lr_norm, dropout_norm, gat_heads_norm, sage_aggr_norm]。"""
    lr_n        = random.random()
    dropout_n   = random.random()
    gat_heads_n = random.random() if 3 in types else 0.0
    sage_aggr_n = random.random() if 4 in types else 0.0
    return [lr_n, dropout_n, gat_heads_n, sage_aggr_n]


def _sample_hp2() -> list:
    """向后兼容：采样旧格式 2 维 HP 向量：[log_lr, dropout]。"""
    log_lr  = random.uniform(LOG_LR_MIN, LOG_LR_MAX)
    dropout = random.uniform(DROPOUT_MIN, DROPOUT_MAX)
    return [log_lr, dropout]


def generate_mini_dataset(
    output_file: str = 'data/mini_gnn_dataset_v5.pkl',
    num_samples: int = 3000,
    skip_prob: float = 0.35,
    hp_dim: int      = 4,
) -> str:
    """
    生成 num_samples 个 v3 空间合法 DAG 样本。

    Parameters
    ----------
    output_file : str    输出 .pkl 文件路径
    num_samples : int    样本数量
    skip_prob   : float  跳跃边生成概率（每条潜在跳跃边独立）
    hp_dim      : int    HP 维度；4=v5 条件参数格式（默认），2=向后兼容
    """
    assert hp_dim in (2, 4), f"hp_dim 必须为 2 或 4，当前: {hp_dim}"
    dataset = []
    print(f"生成 {num_samples} 个 v3 DAG 样本 (max_n=7, nvt=8, hp_dim={hp_dim})...")
    print(f"  主链: 0→1→2→3→4→5→6  每条跳跃边概率={skip_prob}")
    print(f"  REAL_OP_TYPES={REAL_OP_TYPES}  ALL_OP_TYPES={ALL_OP_TYPES}")
    if hp_dim == 4:
        print(f"  HP: [lr_norm, dropout_norm, gat_heads_norm, sage_aggr_norm]")
        print(f"      lr/dropout 全局生效；gat_heads 仅 GATConv 生效；"
              f"sage_aggr 仅 SAGEConv 生效")
    else:
        print(f"  HP: [log_lr ∈ ({LOG_LR_MIN},{LOG_LR_MAX}), "
              f"dropout ∈ ({DROPOUT_MIN},{DROPOUT_MAX})]")

    for _ in tqdm(range(num_samples)):
        # ── 节点类型 ──────────────────────────────────────
        types = [
            START_TYPE,
            random.choice(REAL_OP_TYPES),   # Node1: 禁 Identity
            random.choice(ALL_OP_TYPES),    # Node2: 任意
            random.choice(ALL_OP_TYPES),    # Node3: 任意
            random.choice(ALL_OP_TYPES),    # Node4: 任意
            random.choice(REAL_OP_TYPES),   # Node5: 禁 Identity
            END_TYPE,
        ]

        # ── 邻接矩阵（严格上三角） ────────────────────────
        adj = np.zeros((7, 7), dtype=np.int8)

        # 主链（强制）
        for i in range(6):
            adj[i, i+1] = 1

        # 可选跳跃边: Node_i → Node_j，条件 1 ≤ i < j-1 ≤ 4
        for i in range(1, 5):        # i: 1,2,3,4
            for j in range(i+2, 6):  # j: i+2 .. 5
                if random.random() < skip_prob:
                    adj[i, j] = 1

        # 完整性断言（开发期调试，生产可移除）
        _assert_valid_v3(types, adj)

        # ── HP 采样 ───────────────────────────────────────
        hp = _sample_hp4(types) if hp_dim == 4 else _sample_hp2()
        dataset.append((types, adj, hp))

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\n✅ 数据集保存至: {output_file}  ({len(dataset)} 条，hp_dim={hp_dim})")

    # ── 自检 ──────────────────────────────────────────────
    print("\n[自检] 前 3 条样本：")
    for i, (t, a, hp) in enumerate(dataset[:3]):
        edges = [(r, c) for r in range(7) for c in range(7) if a[r, c]]
        if len(hp) >= 4:
            lr_actual = 10 ** (hp[0] * (LOG_LR_MAX - LOG_LR_MIN) + LOG_LR_MIN)
            drop_actual = hp[1] * (DROPOUT_MAX - DROPOUT_MIN) + DROPOUT_MIN
            hp_str = f"lr={lr_actual:.5f}  drop={drop_actual:.3f}"
            hp_str += f"  gat_heads_n={hp[2]:.3f}  sage_aggr_n={hp[3]:.3f}"
        else:
            hp_str = f"lr={10 ** hp[0]:.5f}  drop={hp[1]:.3f}"
        print(f"  [{i}] types={t}")
        print(f"       edges={edges}  {hp_str}")

    # ── 统计跳跃边比例 ────────────────────────────────────
    n_skip = sum(
        1 for _, a, _ in dataset
        if any(a[i, j] for i in range(1, 5) for j in range(i+2, 6)))
    print(f"\n  含跳跃边样本比例: {n_skip}/{len(dataset)} = "
          f"{n_skip/len(dataset):.1%}")

    # ── GCNII 采样统计 ────────────────────────────────────
    n_gcnii = sum(
        1 for t, _, _ in dataset if 7 in t)
    print(f"  含 GCNII(type=7) 节点样本比例: {n_gcnii}/{len(dataset)} = "
          f"{n_gcnii/len(dataset):.1%}")

    return output_file


def _assert_valid_v3(types, adj):
    """完整性断言，失败时抛出 AssertionError。"""
    # 主链
    for i in range(6):
        assert adj[i, i+1] == 1, f"主链 {i}→{i+1} 缺失"
    # Node 0 只能指向 Node 1
    for j in range(2, 7):
        assert adj[0, j] == 0, f"非法边 0→{j}"
    # Node 6 只能接收来自 Node 5
    for i in range(5):
        assert adj[i, 6] == 0, f"非法边 {i}→6"
    # 禁止反向边
    for i in range(7):
        for j in range(i + 1):
            if i != j:
                assert adj[i, j] == 0, f"反向边 {i}→{j}"
    # Node 1 和 Node 5 不能是 Identity
    assert types[1] != 6, f"Node1 是 Identity，违规！"
    assert types[5] != 6, f"Node5 是 Identity，违规！"
    # 首尾
    assert types[0] == START_TYPE
    assert types[6] == END_TYPE


if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)

    parser = argparse.ArgumentParser(description='generate_mini_data v5')
    parser.add_argument('--output',      type=str,
                        default='data/mini_gnn_dataset_v5.pkl')
    parser.add_argument('--num_samples', type=int,  default=3000)
    parser.add_argument('--skip_prob',   type=float, default=0.35)
    parser.add_argument('--hp_dim',      type=int,  default=4,
                        choices=[2, 4],
                        help='HP 维度：4=v5条件参数格式（默认），2=v3向后兼容')
    parser.add_argument('--seed',        type=int,  default=42)
    cli_args = parser.parse_args()

    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    generate_mini_dataset(
        output_file = cli_args.output,
        num_samples = cli_args.num_samples,
        skip_prob   = cli_args.skip_prob,
        hp_dim      = cli_args.hp_dim,
    )
