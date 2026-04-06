import os
import re

def modernize_code(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 修复标量提取: .data[0] 在新版中会导致 IndexError
    content = re.sub(r'\.data\[0\]', '.item()', content)

    # 2. 移除旧版 Variable 包装 (现代张量原生支持自动求导)
    content = re.sub(r'from torch\.autograd import Variable\n?', '', content)
    # 针对 Variable(x) 或 Variable(x, requires_grad=True) 进行安全脱壳
    content = re.sub(r'Variable\(([^,]+?)\)', r'\1', content)
    content = re.sub(r'Variable\((.+?),\s*requires_grad=True\)', r'\1.requires_grad_()', content)

    # 3. 修复旧版损失函数的 reduction 参数
    content = re.sub(r'size_average=False', "reduction='sum'", content)
    content = re.sub(r'size_average=True', "reduction='mean'", content)
    content = re.sub(r'reduce=False', "reduction='none'", content)

    # 4. 修复掩码的布尔索引 (新版严格要求 bool 类型而非 byte)
    content = re.sub(r'\.byte\(\)', '.bool()', content)
    content = re.sub(r'torch\.ByteTensor', 'torch.BoolTensor', content)

    # 5. 修复旧版 random_ 弃用方法
    content = re.sub(r'\.random_\(2\)', '.random_(0, 2)', content)

    # 覆写原文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Successfully modernized: {filepath}")

if __name__ == "__main__":
    files_to_upgrade = ['train.py', 'models.py', 'util.py']
    for file in files_to_upgrade:
        modernize_code(file)