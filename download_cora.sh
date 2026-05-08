#!/usr/bin/env python3
"""
download_cora.py
================
离线下载 Cora 数据集（PyG Planetoid 格式），支持多镜像自动切换。
不依赖 wget / curl，纯 Python urllib 实现。

用法
----
python download_cora.py              # 默认保存到 /tmp/Cora/raw/
python download_cora.py --root /data/Cora
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import socket

# ============================================================
# 镜像列表（按可达性优先级排序）
# ============================================================
MIRRORS = [
    # Gitee 镜像（国内最稳定）
    "https://gitee.com/jiajiewu/planetoid/raw/master/data",
    # ghproxy 代理
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
    # jsdelivr CDN
    "https://cdn.jsdelivr.net/gh/kimiyoung/planetoid@master/data",
    # GitHub 原址（备用）
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data",
]

FILES = [
    "ind.cora.x",
    "ind.cora.tx",
    "ind.cora.allx",
    "ind.cora.y",
    "ind.cora.ty",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
]


def try_download(url: str, dest: str, timeout: int = 30) -> bool:
    """尝试从 url 下载到 dest，成功返回 True。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 10:          # 文件太小说明是错误页
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except (urllib.error.URLError, socket.timeout, Exception):
        return False


def download_all(root: str):
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"目标目录: {raw_dir}\n")

    failed = []
    for fname in FILES:
        dest = os.path.join(raw_dir, fname)

        if os.path.exists(dest) and os.path.getsize(dest) > 10:
            print(f"  [跳过] {fname}（已存在）")
            continue

        success = False
        for mirror in MIRRORS:
            url = f"{mirror}/{fname}"
            print(f"  下载 {fname} ... 从 {mirror.split('/')[2]}", end="", flush=True)
            if try_download(url, dest):
                size = os.path.getsize(dest)
                print(f"  ✅ ({size:,} bytes)")
                success = True
                break
            else:
                print(f"  ❌ 失败，尝试下一镜像")

        if not success:
            print(f"  ❌❌ {fname} 所有镜像均失败")
            failed.append(fname)

    print()
    if not failed:
        print("=" * 50)
        print("✅ 全部文件下载成功！")
        print(f"   路径: {raw_dir}")
        print("=" * 50)
        print("\n现在可以运行：")
        print(f"  python bo_phase2.py --cora_root {root} ...")
    else:
        print("=" * 50)
        print(f"❌ 以下 {len(failed)} 个文件下载失败：")
        for f in failed:
            print(f"   {f}")
        print()
        print("请在本地有网络的机器执行后 scp 传输：")
        print("  python -c \"from torch_geometric.datasets import Planetoid; "
              "Planetoid(root='/tmp/Cora', name='Cora')\"")
        print(f"  scp -r /tmp/Cora fangxy@202.38.247.36:{root}")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/tmp/Cora")
    a = parser.parse_args()
    download_all(a.root)