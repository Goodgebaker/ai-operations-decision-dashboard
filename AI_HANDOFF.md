# AI 中台运营决策实验台：Windows 迁移与 AI 交接说明

> 本文面向接手本项目的下一位 AI 或开发者。开始修改前，请先完整阅读本文和
> `README.md`，检查 Git 状态，并保留现有五个可见模块、数据口径和测试能力。

## 1. 项目定位

项目名称：**面向 AI 中台的智能运营与动态路由协同优化框架**。

这不是传统 AIOps 监控平台。系统通过真实资源数据校准模拟调用和主动拨测/标准
能力数据，分析模型的运营表现，形成模型能力画像和资源容量结论，为后续多模型
智能路由提供决策依据。

核心决策链路：

```text
真实调用数据 + 主动拨测数据
              ↓
        智能运营分析
              ↓
        模型能力画像
              ↓
        动态智能路由
```

当前项目使用可公开展示的模拟数据、脱敏真实资源汇总和预计算结果，不包含生产
密钥或原始实例 IP。

## 2. 当前仓库与部署

- GitHub：<https://github.com/Goodgebaker/ai-operations-decision-dashboard>
- 在线看板：<https://ai-operations-decision-dashboard-tukdtpg7om3vbegq35b3gt.streamlit.app/>
- 默认分支：`main`
- 当前交接基线提交：`f1202c2`（Fix Streamlit Cloud module imports）
- Streamlit 入口：`dashboard/app.py`
- 推荐 Python：3.11
- 依赖入口：根目录 `requirements.txt`

接手后先执行：

```powershell
git status
git log -5 --oneline
```

如果基线已经有更新，以远端 `main` 的最新提交为准，不要强行回退到上述提交。

## 3. Windows 迁移建议

### 3.1 首选：从 GitHub 克隆

不要复制 macOS 上的 `.venv`、Conda 环境或 Python 缓存。它们包含平台相关二进制，
不能在 Windows 上直接运行。

在 Windows PowerShell 中执行：

```powershell
cd D:\Projects
git clone https://github.com/Goodgebaker/ai-operations-decision-dashboard.git
cd ai-operations-decision-dashboard
```

如果机器无法稳定访问 GitHub，可以先在浏览器下载仓库 ZIP，再按下一节处理。

### 3.2 备选：压缩文件夹复制

可以压缩复制，但应遵循以下规则：

必须保留：

- `dashboard/`
- `src/`
- `data/`
- `outputs/`
- `docs/`
- `tests/`
- `scripts/`
- `.streamlit/config.toml`
- `.github/`
- `.gitignore`
- `requirements.txt`
- `environment.local.yml`
- `README.md`
- `AI_HANDOFF.md`

不要复制或无需复制：

- `.venv/`、`venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.DS_Store`
- 本地日志、临时目录
- `.env`、`.streamlit/secrets.toml` 或任何真实密钥

如果连 `.git/` 一起复制，Git 历史会保留，但当前 Mac 仓库的本地代理配置也可能被
带到 Windows。首次使用 Git 前执行：

```powershell
git config --local --unset-all http.proxy
git config --local --unset-all https.proxy
```

某项不存在时命令可能返回提示，可以忽略。如果 Windows 也需要代理，应改成该电脑
实际使用的代理地址和端口，不要沿用 Mac 的 `127.0.0.1:7897`。

更推荐从 GitHub 重新 `clone`，这样不会继承旧电脑 `.git/config` 中的本地设置。

## 4. Windows 环境安装

### 4.1 使用 Python venv（推荐）

先安装 64 位 Python 3.11 和 Git。然后在项目根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

无需激活环境，直接启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

浏览器访问：<http://localhost:8501>

如果希望激活环境：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard/app.py
```

### 4.2 使用 Conda

```powershell
conda env create -f environment.local.yml
conda activate ai-monitor
python -m streamlit run dashboard/app.py
```

`ARROW_DEFAULT_MEMORY_POOL=system` 主要用于规避原 Intel macOS 环境的 PyArrow
原生层崩溃，在 Windows 上保留通常无害。如果 Windows 出现与内存池相关的错误，
可以新建只包含 Python 3.11 和 `requirements.txt` 依赖的环境进行对照验证。

### 4.3 Windows 不直接使用的文件

- `run_dashboard.sh`
- `scripts/rebuild_demo.sh`
- `scripts/resolve_python.sh`

它们是 Bash 脚本，适用于 macOS/Linux、Git Bash 或 WSL。普通 PowerShell 中应使用
本文给出的 Python 命令。不要因为 `.sh` 不能运行就删除它们，它们仍服务于其他环境。

## 5. 首次验证

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts/check_deployment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本次改造后完整测试数量为 56。后续新增测试后数量可能增加，应以全部通过为准。

再启动页面：

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

依次点击五个左侧模块，确认页面无异常、表格和资源热力图可以显示。

## 6. 五个可见业务模块

1. **运营总览**：调用量、Token、成本、成功率、P95 延迟、模型健康指数及排行。
2. **性能诊断**：P50/P95/P99、延迟波动、稳定性和模型性能评分。
3. **成本分析**：单请求成本、Token 成本、成本趋势、质量/成本比和成本性能评分。
4. **能力校准**：标准任务、真实调用与主动拨测融合诊断、能力画像和路由评分。
5. **资源与容量诊断**：真实并发、等待、TTFT、吞吐、匿名实例 NPU、Cache、HBM
   余量和容量结论。

“智能检测”和“诊断解释”只从页面导航隐藏，`model_health_risk.py`、
`interactive_risk_policy.py`、相关测试和后台产物继续保留。

重要产品原则：

- 页面重点是数据分析、统计评价、模型画像和路由决策支持。
- 不增加无意义装饰，不把系统退化成传统告警大屏。
- 能力校准的主指标是“综合路由评分”，能力/稳定性/性能是分项，可信度是证据可靠性。
- 隐藏模块的检测规则仍来自指标字典，不应因页面隐藏而删除后台兼容能力。
- 成本和能力分没有真实来源，必须在页面标记为模拟假设。
- “Cost Performance Score”在页面上统一显示为“成本性能评分”。

## 7. 关键代码与数据

### 7.1 页面与业务逻辑

- `dashboard/app.py`：Streamlit 五个可见模块页面、交互和可视化入口。
- `src/model_catalog.py`：三个在运模型的统一名称、模拟参数和真实校准入口。
- `src/resource_capacity.py`：每日三表校验、模型级去重、实例脱敏和容量诊断。
- `src/model_operations.py`：运营指标、延迟分位数、成本和健康评分。
- `src/capability_calibration.py`：标准化能力任务与规则评测。
- `src/model_profile.py`：真实调用与主动数据融合诊断、能力画像。
- `src/model_health_risk.py`：健康风险、证据解释、切换和候选模型建议。
- `src/interactive_risk_policy.py`：页面可编辑风险策略、动态事件和未知模式事件。
- `src/model_scoring.py`：统一评分策略、方向、权重和分级。

### 7.2 配置源

`docs/ai_monitoring_metric_dictionary.xlsx` 是指标与规则的主要配置源。重要工作表包括：

- `Scoring Policy`
- `Risk Policy`
- `Diagnosis Policy`
- `Capability Tasks`
- `Active Probes`
- `Probe Assertions`

新增或调整评分时，优先修改配置读取与计算逻辑，不要在多个页面位置重复硬编码权重。

### 7.3 主要输出

- `outputs/model_operating_scores.csv`
- `outputs/model_operating_snapshot.csv`
- `outputs/model_capability_scores.csv`
- `outputs/model_fusion_diagnosis.csv`
- `outputs/model_capability_profiles.csv`
- `outputs/model_health_risks.csv`
- `outputs/model_diagnostic_evidence.csv`
- `data/resource_model_timeseries.csv`
- `data/resource_instance_hourly.csv`
- `outputs/resource_capacity_daily.csv`

当前看板依赖这些预计算文件，因此迁移时必须保留 `data/`、`outputs/` 和指标字典。

三个在运模型固定为 `DeepSeek-V4`、`Minimax-M2.5`、`Qwen3.6-35B-A3B`。
来源工作簿中的“中台模型”不进入看板。

### 7.4 每日真实数据更新

用户每天把以下三份同日期文件放入 `newdata/01_每日三份Excel放这里/`：

- `模型性能中间明细_YYYYMMDD.xlsx`
- `模型性能忙时对比_YYYYMMDD.xlsx`
- `NPU中间统计表_YYYYMMDD.xlsx`

Windows 双击 `更新每日数据.bat`，或执行
`python scripts/update_daily_data.py`。程序会校验、匿名化、按日期替换/追加、归档
原文件、重建产物并运行测试。原始 Excel、归档和 `.instance_salt` 不得提交到 Git。

## 8. 重新生成模拟数据与决策产物

Windows PowerShell 中可按顺序执行：

```powershell
$PYTHON = ".\.venv\Scripts\python.exe"
& $PYTHON src/generate_synthetic_v2.py
& $PYTHON src/build_features.py
& $PYTHON src/composite_rule_engine.py
& $PYTHON src/model_benchmark.py
& $PYTHON src/fusion_rule_engine.py
& $PYTHON src/probe_runner.py
& $PYTHON src/detect_probe_alerts.py
& $PYTHON src/capability_calibration.py
& $PYTHON src/model_operations.py
& $PYTHON src/model_profile.py
& $PYTHON src/model_health_risk.py
```

执行完成后重新运行测试和看板。修改计算逻辑时，必须说明新增指标的计算公式、缺失值
处理、基线窗口、阈值来源和边界行为，并补充测试。

## 9. 已知问题与历史修复

### Streamlit Cloud 找不到 `src`

历史报错：

```text
ModuleNotFoundError: No module named 'src'
```

已在 `dashboard/app.py` 顶部修复：任何 `from src...` 之前先把项目根目录加入
`sys.path`。不要把 `PROJECT_ROOT` 初始化移动到本地模块导入之后。

`tests/test_dashboard_app.py` 中有模拟从 `dashboard/` 目录加载入口的回归测试。

### macOS Streamlit 原生层崩溃

原 Intel Mac 曾发生 `streamlit run dashboard/app.py` segmentation fault，使用
`ARROW_DEFAULT_MEMORY_POOL=system` 和固定环境规避。该问题不代表 Windows 也需要
相同处理，不要在没有复现证据时添加 Windows 专用底层补丁。

### GitHub 连接

原 Mac 使用过本地 Verge 代理 `127.0.0.1:7897`。这属于旧电脑的 `.git/config`
本地设置，不是项目运行依赖。Windows 电脑应使用自身网络配置。

## 10. AI 接手工作规则

下一位 AI 开始工作时必须：

1. 先运行 `git status -sb`，确认是否存在用户未提交的改动。
2. 阅读相关模块、指标字典和现有测试后再修改。
3. 保留用户已有改动，不使用 `git reset --hard` 等破坏性命令。
4. 页面改动不得破坏五个可见模块、左侧大按钮导航及现有数据下载/筛选功能。
5. 业务计算尽量放入 `src/` 并编写单元测试，`dashboard/app.py` 主要负责展示和交互。
6. 新指标必须写清公式、数据来源、窗口、方向、权重、阈值和缺失值处理。
7. 缺少生产数据时，可以使用合理、可复现且明确标注的模拟数据。
8. 修改后至少运行部署检查、完整单元测试和五模块页面冒烟测试。
9. 未经用户明确要求，不自动提交、推送、合并或改动线上部署。
10. 如果要发布，先检查差异，只暂存本次相关文件。

## 11. Git 日常工作流

开始工作前：

```powershell
git switch main
git pull --rebase origin main
git switch -c feature/功能名称
```

修改和验证后：

```powershell
git status
git diff
git add 具体文件1 具体文件2
git diff --cached
git commit -m "简洁说明本次修改"
git push -u origin feature/功能名称
```

随后在 GitHub 创建 Pull Request，检查并合并到 `main`。合并后：

```powershell
git switch main
git pull --rebase origin main
git branch -d feature/功能名称
```

不要不检查就执行 `git add .`，不要把 `.env`、真实密钥、虚拟环境或本地缓存提交到
公开仓库。

## 12. 交接验收标准

完成 Windows 迁移后，应满足：

- Python 3.11 环境可独立重建。
- `scripts/check_deployment.py` 通过。
- 所有单元测试通过。
- `streamlit run dashboard/app.py` 能启动且五个可见模块均可访问。
- 公开 CSV 和看板中不存在原始 `instance/IP`。
- 每日三份 Excel 能通过一键入口完成校验、脱敏和更新。
- `src` 导入无路径错误。
- 指标字典、模拟数据和预计算结果可读取。
- Git 远端指向正确仓库，且不存在旧 Mac 代理配置。
- 没有复制或提交真实密钥。
