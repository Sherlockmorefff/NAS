"""
eval_joint.py
=============
Phase 1 收尾：评估训练好的 JointSpaceVAE 的解码质量。

指标定义
--------
Validity  : 解码出的图是合法 DAG（有 START→END 路径，节点类型合法）的比例
Uniqueness: 合法图中不重复的比例
Novelty   : 合法图中不存在于训练集的比例

跑法
----
python eval_joint.py \
    --checkpoint results/joint_search/joint_model_ep50.pth \
    --data       data/mini_gnn_dataset.pkl \
    --n_samples  1000
"""

import os
import pickle
import argparse
import random
import torch
import igraph
from tqdm import tqdm

from nas_space import JointSpaceVAE

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
                    default='results/joint_search/joint_model_ep50.pth')
parser.add_argument('--data',       type=str,
                    default='data/mini_gnn_dataset.pkl')
parser.add_argument('--n_samples',  type=int, default=1000,
                    help='从隐空间随机采样的数量')
parser.add_argument('--sigma',      type=float, default=1.0,
                    help='采样时的高斯标准差（默认=1.0，即标准正态）')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 工具函数
# ============================================================
class ArchArgs:
    def __init__(self):
        self.max_n           = 5
        self.num_vertex_type = 7
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = 56
        self.bidirectional   = True


def graph_to_hash(g: igraph.Graph) -> str:
    """将 igraph 图转换为可哈希的字符串，用于去重。"""
    types = tuple(g.vs['type'])
    edges = tuple(sorted(g.get_edgelist()))
    return str((types, edges))


def is_valid_dag(g: igraph.Graph,
                 start_type: int = JointSpaceVAE.START_TYPE,
                 end_type:   int = JointSpaceVAE.END_TYPE,
                 num_vertex_type: int = JointSpaceVAE.NUM_VERTEX_TYPE) -> bool:
    """
    合法 DAG 的判定标准：
    1. 是有向无环图（DAG）
    2. 存在且仅存在一个 START 节点（type == start_type）
    3. 存在且仅存在一个 END 节点（type == end_type）
    4. 所有节点 type 在合法范围内
    5. 从 START 到 END 存在有向路径
    """
    # 条件 1：无环
    if not g.is_dag():
        return False

    types = g.vs['type']

    # 条件 4：type 范围合法
    if any(t < 0 or t >= num_vertex_type for t in types):
        return False

    # 条件 2 & 3：START / END 节点数量
    starts = [i for i, t in enumerate(types) if t == start_type]
    ends   = [i for i, t in enumerate(types) if t == end_type]
    if len(starts) != 1 or len(ends) != 1:
        return False

    start_v, end_v = starts[0], ends[0]

    # 条件 5：START → END 可达
    reachable = set(g.subcomponent(start_v, mode='out'))
    if end_v not in reachable:
        return False

    return True


def load_training_hashes(data_path: str) -> set:
    """加载训练集，返回所有图的哈希集合（用于计算 Novelty）。"""
    print(f"加载训练集用于 Novelty 计算: {data_path}")
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)

    hashes = set()
    for types, adj, _ in tqdm(raw_data, desc='哈希训练集'):
        n = len(types)
        g = igraph.Graph(directed=True)
        g.add_vertices(n)
        g.vs['type'] = types
        edges = [(i, j) for i in range(n) for j in range(n)
                 if adj[i][j] == 1 and i < j]
        g.add_edges(edges)
        hashes.add(graph_to_hash(g))
    print(f"训练集共 {len(hashes)} 个唯一图")
    return hashes


# ============================================================
# 主评估逻辑
# ============================================================
def evaluate(model: JointSpaceVAE, n_samples: int, sigma: float,
             train_hashes: set):

    model.eval()
    valid_graphs  = []
    valid_hashes  = []
    n_invalid     = 0

    print(f"\n从隐空间采样 {n_samples} 个点（sigma={sigma}）并解码...")

    with torch.no_grad():
        # 分批解码，避免一次性爆显存
        batch_size = 64
        n_batches  = (n_samples + batch_size - 1) // batch_size

        for _ in tqdm(range(n_batches), desc='解码中'):
            cur_bs   = min(batch_size, n_samples - len(valid_graphs) - n_invalid)
            if cur_bs <= 0:
                break

            z = torch.randn(cur_bs, model.total_nz).to(DEVICE) * sigma
            configs, lrs, drops = model.decode_from_joint_latent(z)

            # configs 里是 gnn_configs dict，但我们需要原始 igraph 对象来做合法性检查
            # 直接从 arch_vae 解码得到图对象
            z_arch  = z[:, :model.arch_nz]
            graphs  = model.arch_vae.decode(z_arch)

            for g in graphs:
                if is_valid_dag(g):
                    valid_graphs.append(g)
                    valid_hashes.append(graph_to_hash(g))
                else:
                    n_invalid += 1

    total_decoded = len(valid_graphs) + n_invalid

    # ---------- 计算三大指标 ----------
    validity   = len(valid_graphs) / total_decoded if total_decoded > 0 else 0.0

    unique_hashes = set(valid_hashes)
    uniqueness  = len(unique_hashes) / len(valid_graphs) if valid_graphs else 0.0

    novel_hashes = unique_hashes - train_hashes
    novelty     = len(novel_hashes) / len(unique_hashes) if unique_hashes else 0.0

    # ---------- 打印报告 ----------
    print("\n" + "=" * 50)
    print("           Phase 1 解码质量评估报告")
    print("=" * 50)
    print(f"  采样总数      : {total_decoded}")
    print(f"  合法图数      : {len(valid_graphs)}")
    print(f"  Validity      : {validity:.2%}   (目标 > 30%，理想 > 70%)")
    print(f"  Uniqueness    : {uniqueness:.2%}  (合法图中不重复的比例)")
    print(f"  Novelty       : {novelty:.2%}   (合法图中训练集未见过的比例)")
    print("=" * 50)

    # ---------- 诊断建议 ----------
    print("\n[诊断]")
    if validity < 0.30:
        print("  ❌ Validity 过低（< 30%）")
        print("     建议：")
        print("     1. 增加训练轮次（EPOCHS 从 50 → 100）")
        print("     2. 检查 D-VAE 的 loss 惩罚项（mu/logvar 范围是否正常）")
        print("     3. 尝试降低采样 sigma（--sigma 0.5），观察是否是隐空间过于弥散")
    elif validity < 0.70:
        print("  ⚠️  Validity 一般（30%~70%）")
        print("     可以进入 Phase 2，但建议继续训练或缩小 sigma 至 0.5~0.8")
    else:
        print("  ✅ Validity 良好（> 70%），可以放心进入 Phase 2")

    if uniqueness < 0.5:
        print("  ⚠️  Uniqueness 偏低：解码器存在模式坍缩，大量采样点映射到同一架构")
        print("     建议：检查 KL 散度权重，考虑加入 beta-VAE 的 beta > 1")

    if novelty < 0.5:
        print("  ⚠️  Novelty 偏低：解码器主要复现训练集，探索能力弱")
        print("     这在 Phase 2 的 BO 阶段会限制搜索范围，留意")

    return {
        'validity':   validity,
        'uniqueness': uniqueness,
        'novelty':    novelty,
        'n_valid':    len(valid_graphs),
        'n_unique':   len(unique_hashes),
        'n_novel':    len(novel_hashes),
    }


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    # 加载模型
    arch_args = ArchArgs()
    model = JointSpaceVAE(arch_args, hp_latent_dim=4).to(DEVICE)

    print(f"加载 checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("模型加载成功 ✅")

    # 加载训练集哈希
    train_hashes = load_training_hashes(args.data)

    # 评估
    results = evaluate(model, args.n_samples, args.sigma, train_hashes)

    # 保存结果
    import json
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至: {out_path}")