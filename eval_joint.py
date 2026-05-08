"""
eval_joint.py  v2  (Search Space v3 对齐)
==========================================
v2 变更（相对 v1）：
  - ArchArgs: max_n=7, num_vertex_type=8, nz=12 对齐 Search Space v3
    （v1 遗留 nz=56 会导致 v3 checkpoint 加载失败）
  - is_valid_dag: NUM_VERTEX_TYPE 从硬编码改为读取 JointSpaceVAE 常量
  - decode_from_joint_latent 调用保持 3 值解包（return_full 默认 False）

用法
----
python eval_joint.py \
    --checkpoint results/joint_search/joint_model_v3_ep100.pth \
    --data       data/mini_gnn_dataset_v4.pkl \
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
                    default='results/joint_search/joint_model_v3_ep100.pth')
parser.add_argument('--data',       type=str,
                    default='data/mini_gnn_dataset_v4.pkl')
parser.add_argument('--n_samples',  type=int, default=1000,
                    help='从隐空间随机采样的数量')
parser.add_argument('--sigma',      type=float, default=1.0,
                    help='采样时的高斯标准差（默认=1.0，即标准正态）')
args = parser.parse_args()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ============================================================
# ArchArgs — v3 对齐（关键修复：v1 中 nz=56 会导致 v3 checkpoint 加载失败）
# ============================================================
class ArchArgs:
    def __init__(self):
        self.max_n           = 7    # v3: 5 → 7
        self.num_vertex_type = 8    # v3: 7 → 8
        self.START_TYPE      = 0
        self.END_TYPE        = 1
        self.hs              = 501
        self.nz              = 12   # v3: 56 → 12（核心修复）
        self.bidirectional   = True


# ============================================================
# 工具函数
# ============================================================
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
    if not g.is_dag():
        return False

    types = g.vs['type']

    if any(t < 0 or t >= num_vertex_type for t in types):
        return False

    starts = [i for i, t in enumerate(types) if t == start_type]
    ends   = [i for i, t in enumerate(types) if t == end_type]
    if len(starts) != 1 or len(ends) != 1:
        return False

    start_v, end_v = starts[0], ends[0]
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
    for item in tqdm(raw_data, desc='哈希训练集'):
        types, adj, _ = item[0], item[1], item[2]
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
        batch_size = 64
        n_batches  = (n_samples + batch_size - 1) // batch_size

        for _ in tqdm(range(n_batches), desc='解码中'):
            cur_bs = min(batch_size, n_samples - len(valid_graphs) - n_invalid)
            if cur_bs <= 0:
                break

            z = torch.randn(cur_bs, model.total_nz).to(DEVICE) * sigma

            # decode_from_joint_latent 默认 return_full=False，保持 3 值解包
            configs, lrs, drops = model.decode_from_joint_latent(z)

            z_arch  = z[:, :model.arch_nz]
            graphs  = model.arch_vae.decode(z_arch)

            for g in graphs:
                if is_valid_dag(g):
                    valid_graphs.append(g)
                    valid_hashes.append(graph_to_hash(g))
                else:
                    n_invalid += 1

    total_decoded = len(valid_graphs) + n_invalid

    validity   = len(valid_graphs) / total_decoded if total_decoded > 0 else 0.0
    unique_hashes = set(valid_hashes)
    uniqueness  = len(unique_hashes) / len(valid_graphs) if valid_graphs else 0.0
    novel_hashes = unique_hashes - train_hashes
    novelty     = len(novel_hashes) / len(unique_hashes) if unique_hashes else 0.0

    print("\n" + "=" * 50)
    print("           Phase 1 解码质量评估报告")
    print("=" * 50)
    print(f"  采样总数      : {total_decoded}")
    print(f"  合法图数      : {len(valid_graphs)}")
    print(f"  Validity      : {validity:.2%}   (目标 > 30%，理想 > 70%)")
    print(f"  Uniqueness    : {uniqueness:.2%}  (合法图中不重复的比例)")
    print(f"  Novelty       : {novelty:.2%}   (合法图中训练集未见过的比例)")
    print("=" * 50)

    print("\n[诊断]")
    if validity < 0.30:
        print("  ❌ Validity 过低（< 30%）")
        print("     建议：增加训练轮次 / 降低采样 sigma / 检查 KL 惩罚项")
    elif validity < 0.70:
        print("  ⚠️  Validity 一般（30%~70%），可进入 Phase 2 但建议继续训练")
    else:
        print("  ✅ Validity 良好（> 70%），可放心进入 Phase 2")

    if uniqueness < 0.5:
        print("  ⚠️  Uniqueness 偏低：解码器存在模式坍缩，考虑加大 beta")

    if novelty < 0.5:
        print("  ⚠️  Novelty 偏低：解码器主要复现训练集，探索能力弱")

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
    arch_args = ArchArgs()
    model = JointSpaceVAE(arch_args, hp_latent_dim=4).to(DEVICE)

    print(f"加载 checkpoint: {args.checkpoint}")
    print(f"  arch_nz={model.arch_nz}  hp_nz={model.hp_nz}  "
          f"total_nz={model.total_nz}")
    state_dict = torch.load(args.checkpoint, map_location=DEVICE,
                            weights_only=True)
    model.load_state_dict(state_dict)
    print("模型加载成功 ✅")

    if not os.path.exists(args.data):
        print(f"⚠️  数据文件不存在: {args.data}，跳过 Novelty 计算")
        train_hashes = set()
    else:
        train_hashes = load_training_hashes(args.data)

    results = evaluate(model, args.n_samples, args.sigma, train_hashes)

    import json
    out_path = os.path.join(os.path.dirname(args.checkpoint), 'eval_results_v2.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存至: {out_path}")