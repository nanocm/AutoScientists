# AutoScientists 本地部署指南（当前机器）

本文档记录 **2026-06-06** 在当前机器上重新搭建并验证 `task-protein-gym` 的实际流程。它替换了上一版中基于旧机器的假设（4× H800、conda 损坏、特定网络限制等）。

当前这次已经验证通过的目标只有一项：

- **重新创建 `task-protein-gym` 环境**
- **确认 ProteinGym Spike 基线可以复现**

---

## 1. 当前机器快照（已验证）

| 项目 | 当前值 |
|------|--------|
| GPU | 2× NVIDIA H800 80GB |
| Driver | 580.82.07 |
| 系统 Python | 3.6.8（过旧，不用于本项目） |
| conda | 23.10.0（可正常使用） |
| 已有可用源环境 | `/opt/conda/envs/python3.10.13` |
| 源环境里的 Torch | `torch 2.8.0`, `cuda=True` |
| 本次新环境 | `/opt/conda/envs/as-protein` |

说明：

- 这台机器上 **conda 已恢复正常**，因此可以正常 `conda create` / `conda clone`。
- 但系统 `python3` 仍然是 **3.6.8**，太旧；运行 `launch.py`、`kermut.py` 等时都应显式使用 conda 环境里的 Python。
- 这次为了最快恢复 ProteinGym 跑通路径，采用的是 **clone 已有 GPU 环境**，而不是从空白环境重装 `torch`。

---

## 2. 创建 ProteinGym 环境（本次实际使用的方法）

本次采用 **clone** 方案：从已经验证可用的 `python3.10.13` 复制出一个专用环境，再补装 ProteinGym 缺少的包。

### 2.1 克隆环境

```bash
conda create -y --name as-protein --clone /opt/conda/envs/python3.10.13
```

### 2.2 安装 ProteinGym 缺少的依赖

当前源环境里已经有：

- `torch 2.8.0`
- `scipy`
- `pandas`
- `numpy`
- `tqdm`
- `requests`
- `pyyaml`

但缺少：

- `gpytorch`
- `h5py`

安装命令：

```bash
/opt/conda/envs/as-protein/bin/python -m pip install gpytorch h5py
```

### 2.3 验证环境

```bash
/opt/conda/envs/as-protein/bin/python - <<'PY'
import importlib
mods=['torch','gpytorch','h5py','scipy','pandas','numpy','tqdm']
for m in mods:
    mod=importlib.import_module(m)
    ver=getattr(mod,'__version__','n/a')
    if m=='torch':
        print(f"{m} {ver} cuda={mod.cuda.is_available()}")
    else:
        print(f"{m} {ver}")
PY
```

本次实测结果：

```text
torch 2.8.0 cuda=True
gpytorch 1.15.2
h5py 3.16.0
scipy 1.14.0
pandas 1.5.3
numpy 1.26.4
tqdm 4.67.3
```

---

## 3. ProteinGym 数据准备

`task-protein-gym` 需要两个数据压缩包：

| 文件 | 用途 |
|------|------|
| `kermut_data.zip` | 官方 kermut 预计算资源（embeddings / ProteinMPNN / coords / zero-shot） |
| `cv_folds_singles_substitutions.zip` | ProteinGym v1.3 的 DMS 数据与 fold 划分 |

### 3.1 本次机器上的现状

本次已经确认以下文件存在：

```text
/root/kermut_data.zip
/root/cv_folds_singles_substitutions.zip
```

并且任务目录下已经有解压后的数据目录：

```text
/root/AutoScientists/task-protein-gym/kermut/data/
```

### 3.2 若需要重新解压

```bash
cd /root/AutoScientists/task-protein-gym
mkdir -p kermut
unzip -q /root/kermut_data.zip -d kermut/
unzip -q /root/cv_folds_singles_substitutions.zip -d kermut/data/
```

最终应包含这些关键路径：

```text
task-protein-gym/kermut/data/
├── conditional_probs/ProteinMPNN/
├── cv_folds_singles_substitutions/
├── embeddings/substitutions_singles/ESM2/
├── structures/coords/
└── zero_shot_fitness_predictions/ESM2/650M/
```

### 3.3 本次验证过的 SPIKE 文件

以下关键输入文件都已确认存在：

```text
kermut/data/cv_folds_singles_substitutions/SPIKE_SARS2_Starr_2020_binding.csv
kermut/data/embeddings/substitutions_singles/ESM2/SPIKE_SARS2_Starr_2020_binding.h5
kermut/data/conditional_probs/ProteinMPNN/SPIKE_SARS2_Starr_2020_binding.npy
kermut/data/structures/coords/SPIKE_SARS2_Starr_2020_binding.npy
kermut/data/zero_shot_fitness_predictions/ESM2/650M/SPIKE_SARS2_Starr_2020_binding.csv
```

---

## 4. 验证 ProteinGym 基线

### 4.1 运行命令

```bash
cd /root/AutoScientists/task-protein-gym
export KERMUT_DATA=$(pwd)/kermut/data
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/as-protein/bin/python \
  repo/kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5
```

### 4.2 本次实测结果

本次重新验证结果：

```text
Overall Spearman: 0.6825
MSE (z-score):    0.581181
```

与 `task-protein-gym/baselines.csv` 中记录的官方复现值一致：

| 指标 | 当前实测 | 仓库基线 |
|------|----------|----------|
| `fold_contiguous_5` | `0.6825` | `0.6820` |
| `mse_fold_contiguous_5` | `0.581181` | `0.5805` |

说明当前机器上的：

- Python 环境
- CUDA / torch
- kermut 数据目录
- `repo/kermut.py`

都已经工作正常。

---

## 5. 启动多智能体 ProteinGym 运行

## 5.1 启动 ClawInstitute

建议在独立 tmux 会话中运行：

```bash
tmux new-session -s claw
npx clawinstitute start
```

默认 API 地址是：

```text
http://localhost:3000/api/v1
```

token 通常会写入：

```text
~/.clawinstitute/token
```

---

## 5.2 创建 run 目录

**不要用系统 `python3`**，应显式使用 conda 环境里的 Python：

```bash
cd /root/AutoScientists
PYTHON=/opt/conda/envs/as-protein/bin/python \
/opt/conda/envs/as-protein/bin/python launch.py spike_v1 --task task-protein-gym
```

这会创建：

```text
/root/spike_v1/
```

---

## 5.3 启动 orchestrator

这一步建议优先使用 **Claude 交互模式**，而不是直接 `claude -p`。

原因：

- `runbook.md + task-profile.md + system/reference/*` 很长
- 交互模式更容易看到过程、处理权限、观察 agent 是否真的开始工作
- 之前在长 prompt 场景里，交互模式比 `claude -p` 更稳

启动方式：

```bash
cd /root/spike_v1
export PYTHON=/opt/conda/envs/as-protein/bin/python
export KERMUT_DATA=/root/AutoScientists/task-protein-gym/kermut/data
claude
```

进入 Claude 提示符后输入：

```text
Read runbook.md and execute.
```

如果你确认当前模型/权限配置已经稳定，也可以尝试 print mode：

```bash
cd /root/spike_v1
PYTHON=/opt/conda/envs/as-protein/bin/python \
KERMUT_DATA=/root/AutoScientists/task-protein-gym/kermut/data \
claude -p "Read runbook.md and execute."
```

但对于排障和首次重新验证，仍推荐优先用交互模式。

---

## 6. 监控与排障

### 6.1 查看 GPU 占用

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
```

### 6.2 查看实验日志

```bash
tail -f /root/spike_v1/logs/experiments.jsonl
```

### 6.3 查看 run 目录关键文件

```bash
ls /root/spike_v1
ls /root/spike_v1/agents
ls /root/spike_v1/logs
```

---

## 7. 当前任务的关键注意事项

1. **不要使用系统 `python3`（3.6.8）**
   - `launch.py`、`repo/kermut.py`、agent 实验脚本都应使用 conda 环境里的 Python。

2. **只从 `KERMUT_DATA` 读取官方数据**
   - 不要改用本地 `task/embeddings_*` 目录。
   - `TASK.md` 和 `LAUNCH.md` 已明确说明：这些本地 embeddings 存在 mutant ordering bug。

3. **ProteinGym 的 split 必须串行运行**
   - 无论手动跑 baseline，还是 GPU agent 提交实验，都不要在同一节点上把三个 split 并行化。

4. **GPU agent 必须串行**
   - `task-protein-gym/LAUNCH.md` 已写明：两个 GPU agent 同时跑会显著拉低结果。

5. **本次文档采用 clone 路线，不是 clean-room 路线**
   - 这是为了最快恢复 `torch+CUDA` 工作环境。
   - 如果以后要写更“标准化、对外可复现”的安装文档，可以再补一版从空白 env 开始的步骤。

---

## 8. 当前已完成状态

截至本次更新，已经确认：

- [x] conda 在新机器上可正常使用
- [x] 成功 clone `/opt/conda/envs/python3.10.13` → `/opt/conda/envs/as-protein`
- [x] 成功安装 `gpytorch` 与 `h5py`
- [x] `task-protein-gym` 所需 SPIKE 数据文件完整
- [x] 基线 `repo/kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5` 成功复现
- [x] 结果为 `Overall Spearman: 0.6825`

下一步就是：

- 启动 `ClawInstitute`
- 创建 `spike_v1`
- 在 `/root/spike_v1` 中启动 orchestrator
