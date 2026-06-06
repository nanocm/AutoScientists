# AutoScientists 工作汇报

**汇报时间：** 2026-06-06  
**覆盖范围：** 上次对话（`2026-06-05-autos.txt`）+ 本次对话全部工作  
**当前主目标：** 在本地机器上跑通并继续推进 `task-protein-gym`，为后续 AutoScientists method 改进与对比实验做准备。

---

## 一、工作总览

本阶段工作主要分为两大部分：

1. **上次对话完成的基础建设工作**
   - 理解论文与代码库结构
   - 梳理三个 benchmark 的定位与依赖
   - 建立本地部署文档与仓库说明
   - 跑通 `task-autoresearch` 与 `task-protein-gym` 的 baseline
   - 初步尝试启动多智能体运行链路

2. **本次对话完成的新机器恢复工作**
   - 清空旧 todo
   - 在新机器上重新检查环境
   - 用 `conda clone` 重建 `protein-gym` 环境
   - 重新验证 `task-protein-gym` baseline
   - 重写 `LOCAL-SETUP.md` 以反映当前机器
   - 提交并推送文档更新
   - 新建 `spike_v2` 运行目录并补发 kickoff post
   - 诊断当前 `claude -p` 后台运行未正常推进的根因

---

## 二、上次对话完成的工作

## 2.1 代码库与论文理解

已完成对仓库整体架构的梳理，核心认知如下：

- **方法实现** 是同一套 AutoScientists 系统：
  - `runbook.md`
  - `launch.py`
  - `system/`
- **任务/benchmark** 是三条不同评测线：
  - `task-autoresearch/`
  - `task-protein-gym/`
  - `task-biomlbench/`

已澄清论文中“方法”和“任务”的关系：

- `AutoScientists` 是本文方法
- `Autoresearch`、`Biomni`、`AIDE` 等是论文里的外部对比方法
- `ProteinGym`、`BioML-Bench`、`GPT training optimization` 是评测任务/benchmark

另外，已对 ProteinGym 任务本身做了详细解释：

- 任务是预测突变对蛋白质功能的影响
- 当前具体子任务是 `SPIKE_SARS2_Starr_2020_binding`
- 目标指标是三种 split 上平均的 Spearman
- baseline 是 `task-protein-gym/repo/kermut.py` 中的 Kermut GP 复现

相关文件：
- `CLAUDE.md`
- `task-protein-gym/TASK.md`
- `task-protein-gym/README.md`
- `task-protein-gym/repo/kermut.py`

---

## 2.2 建立 CLAUDE.md 与本地部署文档

上次对话中完成并提交了：

- `CLAUDE.md`
- `LOCAL-SETUP.md`
- 论文 HTML 文档
- `task-protein-gym/.gitignore`

其中：

### `CLAUDE.md`
整理了仓库的关键运行约定，包括：

- ClawInstitute 的作用与 API 地址
- `launch.py` 与 `runbook.md` 的关系
- 10 个 agent 的角色划分
- `HEARTBEAT.md` 启动分支逻辑
- workspace / workshop / champion / logs 的目录结构
- `PATCH /workspaces/.../files/...` 等关键 API 模式
- 三类任务（`optimization` / `biomlbench` / `proteingym`）的差异
- orchestrator 的系统不变量

### 第一版 `LOCAL-SETUP.md`
记录了旧机器上的部署过程，包括：

- 旧环境特征（4× H800、conda 失效、HF 镜像等）
- `autoresearch` 的两处必要补丁
- `protein-gym` 的手动数据下载与 baseline 验证流程
- 启动多智能体系统的命令建议

---

## 2.3 benchmark 选择、可行性评估与环境排查

上次对话中对三个 benchmark 的工程可行性做了多轮排查。

### 初始判断
- `task-protein-gym`：权威性强，单轮快，适合先做 method 改进
- `task-autoresearch`：最适合作为“只改 orchestration”的对照，但 benchmark 本身学术权威性较弱
- `task-biomlbench`：依赖最重，准备成本最高

### 实测过程中发现的关键现实约束

#### `task-protein-gym`
- `sid.erda.dk` 大文件下载在旧机器上被限速到约 19KB/s
- `marks.hms.harvard.edu` 直连不可达
- 后来通过**用户手动下载** `kermut_data.zip` 和 `cv_folds_singles_substitutions.zip` 解决

#### `task-autoresearch`
- 通过 `hf-mirror.com` 可访问 climbmix 数据
- 上游 `prepare.py` 不自动尊重 `HF_ENDPOINT`
- `train.py` 中默认的 FA3 kernel 路径在旧机器上存在兼容性问题
- 需要把 kernel 固定为 `kernels-community/flash-attn3`
- `kernels` 版本需降到 `0.12.1`

---

## 2.4 跑通 `task-autoresearch` baseline

上次对话中，已为 `task-autoresearch` 做过完整 baseline 打通，主要工作包括：

### 环境准备
- 通过环境复制得到 `as-autoresearch`
- 安装 / 校验：
  - `torch`
  - `pyarrow`
  - `rustbpe`
  - `tiktoken`
  - `kernels==0.12.1`

### 两处关键补丁
1. `task-autoresearch/repo/prepare.py`
   - 将 HuggingFace 数据 URL 改为支持 `HF_ENDPOINT` / `hf-mirror`
2. `task-autoresearch/repo/train.py`
   - 固定使用 `kernels-community/flash-attn3`

### baseline 结果
已成功跑通 `train.py`，得到：

- `val_bpb = 1.014343`
- 训练预算约 300s
- 在 H800 上结果略差于论文 H100 基线是预期现象

这个结果证明：

- 数据准备成功
- tokenizer 可用
- FA3 kernel 可加载
- `train.py` 能在本地 GPU 上跑通

---

## 2.5 跑通 `task-protein-gym` baseline（上次）

在用户手动下载两个关键 zip 后，上次对话已经完成了 `protein-gym` baseline 复现。

### 数据准备
已确认并解压：

- `kermut_data.zip`
- `cv_folds_singles_substitutions.zip`

并验证以下 SPIKE 关键文件存在：

- `cv_folds_singles_substitutions/SPIKE_SARS2_Starr_2020_binding.csv`
- `embeddings/substitutions_singles/ESM2/SPIKE_SARS2_Starr_2020_binding.h5`
- `conditional_probs/ProteinMPNN/SPIKE_SARS2_Starr_2020_binding.npy`
- `structures/coords/SPIKE_SARS2_Starr_2020_binding.npy`
- `zero_shot_fitness_predictions/ESM2/650M/SPIKE_SARS2_Starr_2020_binding.csv`

### baseline 命令
直接运行：

```bash
$PYTHON repo/kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5
```

### 结果
成功复现：

- `Overall Spearman = 0.6825`

对照：
- 仓库记录：`0.6820`

说明 `task-protein-gym` baseline 在本地成功打通。

相关文件：
- `task-protein-gym/repo/kermut.py`
- `task-protein-gym/baselines.csv`

---

## 2.6 多智能体运行链路的首次尝试与问题

上次对话中已经尝试从 baseline 走向完整 AutoScientists 运行。

### 已完成
- 确认/启动过 ClawInstitute
- 使用 `launch.py` 创建过 `spike_v1`
- 补发过 kickoff post
- 验证过 `run_dir`、agent、workspace、workshop 已生成

### 遇到的问题
#### 1. `claude -p` 长 prompt 初始化非常慢
`runbook.md + task-profile.md + system docs` 太长，非交互模式下难以观察真实进展。

#### 2. 插件/初始化问题
有过 `claude-plugins-official` 拉取卡住的现象。

#### 3. 路径/权限问题
曾出现从错误目录启动导致 `/root/spike_v1` 读写受限的情况。

#### 4. 内容策略/API 过滤问题
在 `protein-gym` 场景中，由于涉及：
- `SARS-CoV-2`
- `Spike`
- `ACE2`
- `mutant`
- `fitness`

通过某些代理/模型时较容易触发内容安全过滤。

### 结论
上次对话结束时，已经明确：

- `protein-gym` 的**单脚本 baseline 已经跑通**
- 但**完整 orchestrator 链路尚未稳定跑起来**
- 后续需要在新机器/新模型设置下重新尝试

---

## 三、本次对话完成的工作

## 3.1 清理旧任务上下文

本次对话开始后：

- 已读取并吸收上次完整会话记录：`2026-06-05-autos.txt`
- 已清空旧 todo/task 列表
- 确定当前目标是：
  - 在**新机器**上重新创建环境
  - 更新 `LOCAL-SETUP.md`
  - 重新尝试 `task-protein-gym`

---

## 3.2 新机器环境检查

本次对新机器做了重新盘点，确认：

- `conda 23.10.0` 可正常使用
- 系统 `python3` 仍是 `3.6.8`，不适合跑项目
- 存在可用源环境：
  - `/opt/conda/envs/python3.10.13`
- 该源环境中：
  - `torch 2.8.0`
  - `cuda=True`
- 当前机器 GPU 为：
  - `2× NVIDIA H800 80GB`

同时确认数据文件仍在：

- `/root/kermut_data.zip`
- `/root/cv_folds_singles_substitutions.zip`
- 以及已解压目录：
  - `task-protein-gym/kermut/data/`

---

## 3.3 重建 `task-protein-gym` 环境（本次）

本次明确采用 **clone** 路线，而不是从零新装。

### 原因
- 当前机器上 `conda` 已恢复可用
- `python3.10.13` 中的 `torch + CUDA` 已验证正常
- 目标是尽快恢复 `protein-gym` 跑通链路
- clone 比从空白环境重装 GPU 版 torch 更高效

### 实际执行
成功执行：

```bash
conda create -y --name as-protein --clone /opt/conda/envs/python3.10.13
```

随后补装 ProteinGym 缺失依赖：

```bash
/opt/conda/envs/as-protein/bin/python -m pip install gpytorch h5py
```

### 最终验证
环境中确认存在：

- `torch 2.8.0`（`cuda=True`）
- `gpytorch 1.15.2`
- `h5py 3.16.0`
- `scipy 1.14.0`
- `pandas 1.5.3`
- `numpy 1.26.4`
- `tqdm 4.67.3`

---

## 3.4 在新机器上重新验证 `protein-gym` baseline

本次在新环境 `as-protein` 中重新运行 baseline：

```bash
cd /root/AutoScientists/task-protein-gym
export KERMUT_DATA=$(pwd)/kermut/data
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/as-protein/bin/python \
  repo/kermut.py SPIKE_SARS2_Starr_2020_binding fold_contiguous_5
```

### 结果
成功复现：

- `Overall Spearman: 0.6825`
- `MSE (z-score): 0.581181`

对照 `task-protein-gym/baselines.csv`：

- `fold_contiguous_5_repo_kermut = 0.6820`
- `mse_fold_contiguous_5_repo_kermut = 0.5805`

说明在**新机器**上：

- 复制后的 conda 环境可用
- 数据目录无问题
- `repo/kermut.py` 运行正常
- baseline 成功复现

---

## 3.5 重写 `LOCAL-SETUP.md`

本次已将 `LOCAL-SETUP.md` 改写为**当前机器版本**，重点修正了上一版与旧环境绑定过深的问题。

### 新文档内容要点
- 当前机器真实快照（2× H800、conda 可用、系统 python 3.6.8 等）
- 使用 `conda clone` 构建 `as-protein` 的步骤
- 本机上 `protein-gym` 数据文件与解压目录现状
- 基线验证命令与结果
- `ClawInstitute` 启动方式
- 建议优先用交互式 `claude` 启动 orchestrator
- 当前阶段只聚焦 `task-protein-gym`

### 注意
这一版 `LOCAL-SETUP.md` 已不再假设：
- 旧机器的 conda 崩坏状态
- 旧机器的 4× H800
- 那套旧环境专属 workaround 必然成立

---

## 3.6 commit / push

本次已完成代码提交与推送。

### 过程
1. 先在分支 `update-local-setup-proteingym` 上提交：
   - commit: `0d37079`
   - message: `Update local setup for current protein-gym machine`
2. 根据要求将其 **fast-forward 合并到 `main`**
3. 已 push 到：
   - `origin/main`

### 处理细节
- 未把无关未跟踪文件加入提交：
  - `2026-06-05-autos.txt`
  - `*.excp`
- 顺手修正了 `task-autoresearch/download_repo.sh` 的执行权限漂移，避免污染这次提交

---

## 3.7 新建 `spike_v2` 并准备多智能体运行

为避免复用旧的 `spike_v1` 状态，本次新建了一个干净 run：

- `/root/spike_v2`

### 实际执行
使用：

```bash
PYTHON=/opt/conda/envs/as-protein/bin/python \
/opt/conda/envs/as-protein/bin/python launch.py spike_v2 --task task-protein-gym
```

### 创建结果
成功完成：
- workshop：`ar_spike_v2`
- workspace ID：`4b703ae1-a53f-484d-a4bf-3d43364ac872`
- 10 个 agent 注册
- `task-profile.md` 复制完成
- `runbook.md` / `task/` / `repo/` / `system/` 生成完成

### kickoff post 问题与修复
`launch.py` 自带的 kickoff post 仍出现历史老问题：
- HTTP 400
- 原因是 post 字段名/调用格式与服务端不一致

已手动使用 agent token 补发成功：
- 标题：`[KICKOFF] ProteinGym Spike run initialized`

说明：
- 现在 `spike_v2` 的 workshop 已具备最基本讨论入口

---

## 3.8 ClawInstitute 状态确认

本次再次确认了本地协调服务状态：

- 端口 `3000` 已有服务
- API 可返回 `HTTP 200`
- token 可用

在尝试重复 `npx clawinstitute start` 时收到：
- `EADDRINUSE`

这说明：
- 不是没启动
- 而是**本地服务已经在运行**

---

## 3.9 `claude -p` / auto mode / classifier 依赖关系澄清

本次对 Claude Code 的运行机制做了澄清，结论如下：

### 事实
- `claude -p` 是 **print / 非交互模式**
- 它本身**不等于** auto mode
- 但如果其工具执行走的是 auto mode 权限策略，那么 `Bash` 等动作仍要依赖 Claude classifier

### 已观察到的现象
在当前环境中已明确出现：

- `claude-opus-4-8[1m] is temporarily unavailable, so auto mode cannot determine the safety of Bash right now`

这说明：
- 即便主任务想切到 GPT 或别的 provider
- **Claude Code auto mode 的工具安全分类器仍然依赖 Claude/Opus 侧能力**

### 影响
这会直接影响：
- `claude -p` 运行 runbook
- 交互式 `claude` 里的某些 `Bash` 工具调用
- 尤其是长流程 orchestrator 场景

---

## 3.10 当前 `claude -p "Read runbook.md and execute."` 状态诊断

本次最后还检查了你后台执行的 `claude -p` 是否正常运行。

### 结论
**它并没有正常推进。**

### 证据
1. 后台确实有 `claude -p` 进程在跑
2. `spike_v2` 的 workshop 中仍然只有 1 条 kickoff post
3. `spike_v2/logs/` 下没有新的执行日志
4. tmux 输出显示它在读完部分 runbook / task-profile / agent 文件后，进入需要执行关键 `Bash(...)` 的阶段时，被 auto mode classifier 卡住

典型报错：
- `auto mode cannot determine the safety of Bash right now`
- `Interrupted · What should Claude do instead?`

### 当前判断
因此当前状态应理解为：

- **进程已启动**
- **但没有进入真正的 orchestrator 正常执行态**
- **当前阻塞点是 auto mode / classifier，而不是 task 本身或数据本身**

---

## 四、当前产出物清单

本阶段已形成/更新的关键产出：

### 文档
- `CLAUDE.md`
- `LOCAL-SETUP.md`（旧版 + 当前机器重写版）
- `WORK-REPORT.md`（本汇报）

### benchmark / setup 相关
- `task-protein-gym/repo/kermut.py` baseline 跑通
- `task-autoresearch/repo/train.py` baseline 跑通（上次对话）
- `task-autoresearch/repo/prepare.py` / `train.py` 的兼容性补丁方案已验证（上次对话）

### 运行目录
- `spike_v1/`（旧尝试）
- `spike_v2/`（当前建议继续使用的干净 run）

### Git 状态
- `main` 已包含当前机器版 `LOCAL-SETUP.md` 更新

---

## 五、当前已确认的关键事实

1. **`task-protein-gym` baseline 在新机器上已重新成功复现**
   - 当前最可靠、最明确的事实

2. **当前 `as-protein` 环境可直接复用**
   - `/opt/conda/envs/as-protein/bin/python`

3. **`spike_v2` 运行目录已完成初始化**
   - workshop / workspace / agents / task-profile 均已存在

4. **当前主要阻塞不在 benchmark 本身，而在 orchestrator 启动方式**
   - 尤其是 `claude -p` + auto mode classifier 这一层

5. **直接用 `claude -p` 跑长 runbook 不稳定**
   - 在当前机器/当前模式下已再次得到验证

---

## 六、遗留问题 / 风险点

## 6.1 orchestrator 启动仍未稳定跑通
`spike_v2` 已初始化，但完整多智能体循环仍未开始。

### 当前主要原因
- auto mode classifier 对 `Bash` 的判定依赖 Claude Opus 可用性
- 当 classifier unavailable 时，runbook 执行会在关键 Bash 步骤停住

---

## 6.2 `launch.py` 的 kickoff post 仍有兼容性问题
虽然本次已手工补发，但从工程上看：
- `launch.py` 自动 kickoff 仍未真正修复
- 后续如果频繁创建新 run，仍会重复遇到

---

## 6.3 本地仍有未清理文件
当前仓库外观上还有未跟踪文件：
- `2026-06-05-autos.txt`
- `*.excp`

它们未影响本次提交，但后续可做清理。

---

## 七、建议的下一步

## 优先级 1：不要再依赖 `claude -p` 直接跑完整 runbook
建议改用：

```bash
cd /root/spike_v2
export PYTHON=/opt/conda/envs/as-protein/bin/python
export KERMUT_DATA=/root/AutoScientists/task-protein-gym/kermut/data
claude
```

然后在交互界面输入：

```text
Read runbook.md and execute.
```

原因：
- 更容易观察它卡在哪一步
- 遇到权限/classifier问题时更可恢复
- 比长时间黑盒 `claude -p` 更适合当前状态

---

## 优先级 2：为 `spike_v2` 配一套更合适的 Claude Code 允许策略
目标：减少这些操作被 classifier 阻塞：
- `python ...`
- `bash ...`
- `find ...`
- `grep ...`
- `curl ...`
- 本地只读检查命令

如果后续继续依赖 Claude Code 来驱动 runbook，这一步很重要。

---

## 优先级 3：如果继续做 method 改进实验
建议顺序是：

1. 先让 `spike_v2` 的原版 AutoScientists 跑起来
2. 建立一条原版轨迹
3. 再选定一个 method 改进点做 A/B

当前适合的改进点候选：
- 团队形成 / 自组织机制
- 讨论与 proposal 审核机制
- 负知识共享方式
- GPU 调度与队列管理
- 显式多样性激励
- meta-improvement 触发与编辑策略

---

## 八、阶段结论

截至当前：

- **基础设施层面**：已完成文档建设、仓库说明整理、环境重建、基线复现、run 目录生成
- **实验准备层面**：`task-protein-gym` 已经具备继续做多智能体实验的条件
- **主要阻塞层面**：当前的核心问题已经从“环境/数据/benchmark 能不能跑”转移为“Claude Code runbook 驱动如何稳定执行”

换言之：

> **ProteinGym baseline 已经在新机器上重新完全打通；现在真正需要解决的是 orchestrator 的执行稳定性，而不是任务本身。**

---

## 九、附录：本阶段关键路径摘要

### 已成功路径
- 新机器 conda 正常
- clone `python3.10.13` → `as-protein`
- 安装 `gpytorch` / `h5py`
- 复现 `protein-gym` baseline `0.6825`
- 重写 `LOCAL-SETUP.md`
- commit + push 到 `main`
- 创建 `spike_v2`
- 手动补发 kickoff post

### 当前失败 / 阻塞路径
- `claude -p "Read runbook.md and execute."`
  - 进程存在
  - 但在需要 auto-classified `Bash` 时被阻断
  - 尚未真正推进到 analyst / GPU agent 实验阶段
