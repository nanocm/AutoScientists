# AutoScientists 本地部署指南

本文档记录在当前 H800 平台上部署和运行 AutoScientists 的完整流程，包括环境搭建、数据准备、基线验证和多智能体系统启动。

## 硬件与网络环境

| 项目 | 配置 |
|------|------|
| GPU | 4× NVIDIA H800 80GB |
| CUDA | 13.0, Driver 580.105.08 |
| OS | Linux 5.10 (Alibaba Cloud, glibc 2.32) |
| Python | 系统 3.6（不可用）; conda env 3.10.13 |
| 网络 | github.com ✅ / huggingface.co ❌ 需用 `hf-mirror.com` / marks.hms.harvard.edu ❌ / sid.erda.dk 限速 19KB/s |

## 前置依赖

- **Node.js 22+**（运行 ClawInstitute 协调服务器）
- **conda**（注意：本机 base conda 的 urllib3 损坏，`conda create` 不可用，需用 `cp -a` 克隆环境）
- **Claude Code CLI**（`claude` 命令）

## 一、Python 环境

由于 base conda 损坏（`from collections import Mapping` 在 Python 3.10 上崩溃），使用目录拷贝方式创建独立环境：

```bash
# 从已有 conda env 克隆（不要用 conda create --clone，会崩）
cp -a /opt/conda/envs/python3.10.13 /opt/conda/envs/as-autoresearch

# 注意：克隆后 bin/pip 的 shebang 仍指向源环境，必须用 python -m pip
PYTHON=/opt/conda/envs/as-autoresearch/bin/python

# 安装 AutoScientists 基础依赖
$PYTHON -m pip install requests pyyaml

# task-protein-gym 依赖
$PYTHON -m pip install gpytorch h5py scipy pandas numpy tqdm

# task-autoresearch 依赖
$PYTHON -m pip install pyarrow rustbpe tiktoken "kernels==0.12.1"
# kernels 必须 0.12.1 — 0.15+ 的 API 变更导致 get_kernel() 需要 version 参数，与上游 train.py 不兼容
```

验证环境：

```bash
$PYTHON -c "
import torch, gpytorch, h5py, kernels, rustbpe, tiktoken, pyarrow
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('gpytorch', gpytorch.__version__)
print('kernels', kernels.__version__)
print('All OK')
"
```

## 二、启动 ClawInstitute 协调服务器

所有多智能体运行都依赖此服务器（HTTP API 在 `localhost:3000`）。建议在独立 tmux 会话中运行：

```bash
tmux new-session -s claw
npx clawinstitute start
# 首次运行会从 npm 下载，后续使用缓存
# Ctrl-B D 可脱离 tmux
```

token 自动写入 `~/.clawinstitute/token`，`launch.py` 会自动读取。

## 三、task-protein-gym（ProteinGym Spike 适应度预测）

### 3.1 数据准备

本机无法自动下载数据（ERDA 限速 19KB/s，Harvard 服务器不通），需**手动下载**以下两个文件放到 `~/`：

| 文件 | 来源 | 大小 |
|------|------|------|
| `kermut_data.zip` | https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip | 3.8 GB |
| `cv_folds_singles_substitutions.zip` | https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/cv_folds_singles_substitutions.zip | 13 MB |

解压到任务目录：

```bash
cd /root/AutoScientists/task-protein-gym
mkdir -p kermut
unzip -q ~/kermut_data.zip -d kermut/
unzip -q ~/cv_folds_singles_substitutions.zip -d kermut/data/
```

最终目录结构：

```
task-protein-gym/kermut/data/
├── conditional_probs/ProteinMPNN/       # ProteinMPNN AA 分布 (.npy)
├── cv_folds_singles_substitutions/      # DMS 数据 + fold 列 (.csv)
├── embeddings/substitutions_singles/ESM2/ # ESM-2 embeddings (.h5)
├── structures/coords/                   # Cα 三维坐标 (.npy)
└── zero_shot_fitness_predictions/       # ESM-2 zero-shot 分数 (.csv)
```

### 3.2 验证基线

```bash
PYTHON=/opt/conda/envs/as-autoresearch/bin/python
KERMUT_DATA=/root/AutoScientists/task-protein-gym/kermut/data

CUDA_VISIBLE_DEVICES=0 $PYTHON task-protein-gym/repo/kermut.py \
  SPIKE_SARS2_Starr_2020_binding fold_contiguous_5
```

预期结果：`Overall Spearman: 0.6825`（与仓库基线 0.6820 在单种子方差内吻合）。

三个 split 的复现基线：

| Split | Spearman | 仓库基线 |
|-------|----------|----------|
| fold_contiguous_5 | 0.6825 | 0.6820 |
| fold_modulo_5 | — | 0.7042 |
| fold_random_5 | — | 0.8423 |
| **均值** | — | **0.743** |

### 3.3 启动多智能体运行

```bash
# 1. 创建实验目录（仅需一次）
cd /root/AutoScientists
$PYTHON launch.py spike_v1 --task task-protein-gym

# 2. 在新 tmux 窗口中启动 orchestrator
tmux new-session -s orchestrator

PYTHON=/opt/conda/envs/as-autoresearch/bin/python \
KERMUT_DATA=/root/AutoScientists/task-protein-gym/kermut/data \
HF_ENDPOINT=https://hf-mirror.com \
claude -p "Read /root/spike_v1/runbook.md and execute. Task: task-protein-gym. Run name: spike_v1."
```

Orchestrator 检测到 `spike_v1/WORKSPACE_ID` 已存在后直接进入执行循环，不会重复建目录。运行数小时直到所有 GPU agent 用完 10 次提交配额。

## 四、task-autoresearch（nanoGPT val_bpb 优化）

### 4.1 代码准备

```bash
# 克隆上游仓库（仅需一次）
bash task-autoresearch/download_repo.sh
```

### 4.2 必须打的两个补丁

由于网络和 glibc 限制，`task-autoresearch/repo/` 中需要两处修改。下方是已打过的补丁内容，供核对/重做：

**补丁 1：`prepare.py` — HuggingFace 镜像**

原代码：
```python
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
```

改为：
```python
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
BASE_URL = f"{_HF_ENDPOINT}/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
```

原因：`prepare.py` 用的是裸 `requests.get` 而非 `huggingface_hub`，`HF_ENDPOINT` 环境变量不会自动生效，必须手动替换 URL。

**补丁 2：`train.py` — FA3 Kernel 兼容**

原代码：
```python
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
```

改为：
```python
repo = "kernels-community/flash-attn3"
```

原因：`varunneal/flash-attention-3` 的预编译二进制需要 glibc 2.34，本机是 2.32，加载时报 `GLIBC_2.34 not found`。`kernels-community/flash-attn3` 在 glibc 2.32 上正常工作。

### 4.3 数据准备

```bash
cd /root/AutoScientists/task-autoresearch/repo

# 下载训练数据（15 shards + 1 val shard, ~1.4GB）+ 训练 BPE tokenizer
HF_ENDPOINT=https://hf-mirror.com $PYTHON prepare.py --num-shards 15
```

数据缓存在 `~/.cache/autoresearch/`（data/ + tokenizer/）。

FA3 kernel 首次加载时自动缓存到 `/tmp/huggingface/hub/`。

### 4.4 验证基线

```bash
cd /root/AutoScientists/task-autoresearch/repo

HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0 \
$PYTHON train.py
```

预期结果（5 分钟训练预算，H800）：

| 指标 | 值 |
|------|------|
| val_bpb | 1.014 |
| training_seconds | 300 |
| num_steps | 730 |
| num_params_M | 50.3 |
| mfu_percent | 30.4% |
| peak_vram | 45 GB |

论文基线是 val_bpb≈0.998（H100），差异来自 H800 在固定 300s 预算内步数更少。A/B 对比看相对 delta，绝对偏移不影响结论。

### 4.5 启动多智能体运行

```bash
cd /root/AutoScientists
$PYTHON launch.py ar_v1 --task task-autoresearch

tmux new-session -s orchestrator-ar

HF_ENDPOINT=https://hf-mirror.com \
claude -p "Read /root/ar_v1/runbook.md and execute. Task: task-autoresearch. Run name: ar_v1."
```

## 五、监控运行中的实验

```bash
# 实时观察实验结果
tail -f <run-dir>/logs/experiments.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f'{d.get(\"completed_at\",\"?\")} {d.get(\"exp_id\",\"?\"):40s} {d.get(\"outcome\",\"?\")} delta={d.get(\"delta\",0):+.6f}')
"

# KEEP 率统计
cat <run-dir>/logs/experiments.jsonl | python3 -c "
import sys, json
from collections import defaultdict
stats = defaultdict(lambda: {'keep': 0, 'total': 0})
for line in sys.stdin:
    d = json.loads(line)
    t = d.get('team', 'unknown')
    stats[t]['total'] += 1
    if d.get('outcome') == 'KEEP': stats[t]['keep'] += 1
for t, s in stats.items():
    print(f'{t}: {s[\"keep\"]}/{s[\"total\"]} KEEPs ({100*s[\"keep\"]/max(s[\"total\"],1):.0f}%)')
"

# ClawInstitute 工作区文件列表
TOKEN=$(python3 -c "import json; print(list(json.load(open('<run-dir>/agent_tokens.json')).values())[0])")
WS=$(cat <run-dir>/WORKSPACE_ID)
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:3000/api/v1/workspaces/$WS/files" | python3 -m json.tool
```

## 六、已知问题与注意事项

1. **不要用 `conda create`** — base conda 的 urllib3 损坏，任何 conda 命令都会崩溃。用 `cp -a` 克隆环境 + `python -m pip` 安装包。

2. **不要用 `task/embeddings_*/` 目录** — SPIKE embeddings 的 mutant ordering 有 bug（3800/3802 行错位），会导致静默的特征-标签错配和虚高的 Spearman 分数。始终从 `KERMUT_DATA` 的 h5 文件加载。

3. **GPU agent 必须串行** — protein-gym 任务中两个 GPU agent 并行会导致 fold_contiguous_5 从 0.68 降到 0.54（GPU 争用）。

4. **FA3 kernel 缓存在 `/tmp/`** — 如果 `/tmp/` 被清理，下次 `train.py` 启动会重新从 hf-mirror 下载（约 3 分钟）。

5. **Analyst agent 不要用 haiku 模型** — 有文档化的失败模式：haiku 会写本地 memory 文件声称已完成但从不调用 ClawInstitute API，导致队列无法被填充。始终用 sonnet 或 opus。

6. **HF_ENDPOINT 需要显式传递** — 环境变量 `HF_ENDPOINT=https://hf-mirror.com` 对 `huggingface_hub` 库生效，但 autoresearch 的 `prepare.py` 用的是裸 `requests.get`，已通过补丁 1 处理。`kernels` 库（FA3 下载）会自动尊重此变量。
