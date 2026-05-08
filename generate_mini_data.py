"""
generate_mini_data.py  v3
=========================
Search Space v3: max_n=7 (5 算子节点), nvt=8 (新增 GCNII=type 7)

节点编号 (7 节点, 索引 0~6)：
  0 = START
  1 = Node1 (算子, 禁止 Identity)
  2 = Node2 (算子/Identity)
  3 = Node3 (算子/Identity)
  4 = Node4 (算子/Identity)
  5 = Node5 (算子, 禁止 Identity)
  6 = END

合法拓扑规则：
  ① 主链: 0→1→2→3→4→5→6
  ② Node 0 只能指向 Node 1
  ③ Node 6 只能接收来自 Node 5 的边
  ④ 可选跳跃边: Node_i → Node_j (i < j-1, i≥1, j≤5)
  ⑤ Node 1 和 Node 5 禁止 Identity(6)
  ⑥ Node 2/3/4 可以是任意算子（含 Identity）
"""

import pickle
import random
import numpy as np
from tqdm import tqdm

START_TYPE    = 0
END_TYPE      = 1
# v3: 新增 GCNII = 7
REAL_OP_TYPES = [2, 3, 4, 5, 7]   # GCNConv, GATConv, SAGEConv, GINConv, GCNII
ALL_OP_TYPES  = [2, 3, 4, 5, 6, 7] # 上述 + Identity


def generate_mini_dataset(
    output_file: str = 'data/mini_gnn_dataset_v3.pkl',
    num_samples: int = 3000,
    skip_prob: float = 0.35,
):
    """
    生成 num_samples 个 v3 空间合法 DAG 样本 (7 节点)。

    Parameters
    ----------
    skip_prob : float
        每条潜在跳跃边 (i→j, i<j-1) 独立添加的概率
    """
    dataset = []
    print(f"生成 {num_samples} 个 v3 DAG 样本 (max_n=7, nvt=8)...")
    print(f"  主链: 0→1→2→3→4→5→6  每条跳跃边概率={skip_prob}")
    print(f"  REAL_OP_TYPES={REAL_OP_TYPES}  ALL_OP_TYPES={ALL_OP_TYPES}")

    for _ in tqdm(range(num_samples)):
        # ── 节点类型 ──────────────────────────────────────────
        types = [
            START_TYPE,
            random.choice(REAL_OP_TYPES),   # Node1: 禁 Identity
            random.choice(ALL_OP_TYPES),    # Node2: 任意
            random.choice(ALL_OP_TYPES),    # Node3: 任意
            random.choice(ALL_OP_TYPES),    # Node4: 任意
            random.choice(REAL_OP_TYPES),   # Node5: 禁 Identity
            END_TYPE,
        ]

        # ── 邻接矩阵（严格上三角）──────────────────────────────
        adj = np.zeros((7, 7), dtype=np.int8)

        # 主链（强制）
        for i in range(6):
            adj[i, i+1] = 1

        # 可选跳跃边: Node_i → Node_j，条件 1 ≤ i < j-1 ≤ 4
        for i in range(1, 5):        # i: 1,2,3,4
            for j in range(i+2, 6):  # j: i+2 .. 5
                if random.random() < skip_prob:
                    adj[i, j] = 1

        _assert_valid_v3(types, adj)
        log_lr  = random.uniform(-4.0, -1.5)
        dropout = random.uniform(0.1, 0.6)
        dataset.append((types, adj, [log_lr, dropout]))

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\n✅ 数据集保存至: {output_file}  ({len(dataset)} 条)")

    print("\n[自检] 前 3 条样本：")
    for i, (t, a, hp) in enumerate(dataset[:3]):
        edges = [(r, c) for r in range(7) for c in range(7) if a[r, c]]
        print(f"  [{i}] types={t}")
        print(f"       edges={edges}  lr={10**hp[0]:.5f}  drop={hp[1]:.3f}")

    n_skip = sum(1 for _, a, _ in dataset
                 if any(a[i, j] for i in range(1,5) for j in range(i+2,6)))
    print(f"\n  含跳跃边样本比例: {n_skip}/{len(dataset)} = {n_skip/len(dataset):.1%}")
    return output_file


def _assert_valid_v3(types, adj):
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
        for j in range(i+1):
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
    generate_mini_dataset(
        output_file='data/mini_gnn_dataset_v3.pkl',
        num_samples=3000,
        skip_prob=0.35,
    )