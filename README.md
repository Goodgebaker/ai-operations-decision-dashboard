# AI 中台运营决策实验台

这是一个面向多模型智能路由的运营决策原型：融合真实调用、主动可用性拨测和
标准能力校准数据，形成模型运营评分、能力画像、健康风险、诊断解释与路由建议。
原有复合规则、滚动 MAD、STL 和 Isolation Forest 异常检测能力继续保留。

项目内置的是可公开展示的模拟数据与预计算结果，不包含生产凭据。看板入口为
`dashboard/app.py`，支持从项目根目录本地运行，也可直接部署到 Streamlit
Community Cloud。

## 快速开始

项目使用 Python 3.11。首次安装推荐使用项目的本地 Conda 配置：

```bash
conda env create -f environment.local.yml
conda activate ai-monitor
./run_dashboard.sh
```

如果环境已经存在，更新依赖即可：

```bash
conda env update -f environment.local.yml --prune
conda activate ai-monitor
./run_dashboard.sh
```

也可以使用标准虚拟环境和 pip：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_dashboard.sh
```

浏览器访问 `http://localhost:8501`。`run_dashboard.sh` 会先切换到项目根目录，
并依次查找项目 `.venv`、已激活的 `ai-monitor` 环境或同名 Conda 环境，因此也
可以从其他目录调用，并且不会误用系统 Python。

## 公网部署

仓库已经按 Streamlit Community Cloud 的约定整理：根目录只有
`requirements.txt` 会被云端识别为依赖入口；`environment.local.yml` 只服务于
本地 Conda 环境。部署前先执行：

```bash
python scripts/check_deployment.py
python -m unittest discover -s tests -v
```

将仓库推送到 GitHub 后，在 `https://share.streamlit.io/` 创建应用并填写：

- Repository：你的 GitHub 仓库
- Branch：`main`
- Main file path：`dashboard/app.py`
- Python version：`3.11`

部署完成后会得到可公开访问的 `*.streamlit.app` 地址。GitHub Actions 会在每次
推送和 Pull Request 时检查部署资产、编译源码并运行测试；Dependabot 每周检查
Python 与 GitHub Actions 依赖更新。

当前看板读取仓库中的模拟 CSV 和指标字典，因此 `data/`、`outputs/` 中被应用
引用的文件以及 `docs/ai_monitoring_metric_dictionary.xlsx` 必须进入 Git。真实
API Key 不得写入仓库；实拨所需变量名可以参考 `.env.example`，真实值应由运行
环境的 Secrets 或环境变量管理。

## 在 VS Code 中运行

先打开项目文件夹，在 VS Code 右下角选择 `ai-monitor` Python 环境。随后打开“终端 → 新建终端”，确认终端前缀是 `(ai-monitor)`。

项目在 Intel macOS 上固定使用 PyArrow 的系统内存池，避免 PyArrow 25 的
`mimalloc` 在多线程字符串去重时触发原生层段错误。该变量已记录在
`environment.local.yml`，并由 `run_dashboard.sh` 在 Streamlit 导入前再次保证。

按顺序执行：

```bash
python src/generate_synthetic_v2.py
python src/build_features.py
python src/composite_rule_engine.py
python src/model_benchmark.py
python src/fusion_rule_engine.py
python src/probe_runner.py
python src/detect_probe_alerts.py
python src/capability_calibration.py
python src/model_operations.py
python src/model_profile.py
python src/model_health_risk.py
streamlit run dashboard/app.py
```

也可以一次性重建全部模拟数据与决策产物：

```bash
./scripts/rebuild_demo.sh
```

也可以使用推荐的安全启动入口：

```bash
./run_dashboard.sh
```

如果刚为现有 Conda 环境写入环境变量，需要先执行一次：

```bash
conda deactivate
conda activate ai-monitor
```

之后原来的 `streamlit run dashboard/app.py` 也会自动使用系统内存池。

其中 `probe_runner.py` 默认生成 30 天主动拨测模拟记录，便于在没有真实接口凭据时验证告警流程。接入真实接口后，可通过专用环境变量配置 endpoint 和 API Key，再执行一次拨测或启动定时调度：

```bash
python src/probe_scheduler.py --once
# 或持续运行
python src/probe_scheduler.py
```

实时拨测采用独立的 `traffic_type=probe`，不会混入真实用户调用。具体任务、频率、断言、超时和密钥变量名都在指标字典的 `Active Probes`、`Probe Assertions` 工作表中维护。

标准能力校准由 `Capability Tasks` 工作表维护。运行 `capability_calibration.py`
会让每个启用模型执行完全相同的任务、重复次数和评测规则，模拟模式默认生成
30 天对称样本。能力数据固定标记为 `traffic_type=capability_probe`，不会混入
真实调用或现有可用性拨测数据。

`model_operations.py` 使用原始调用日志计算准确的日级 P50/P95/P99、成功率和
成本指标，使用模型小时特征计算日内波动，再按 `Scoring Policy` 生成性能、
稳定性、成本效率、成本性能和健康指数。成本趋势以单请求成本相对前 7 个
历史日中位数计算，至少需要 3 个历史日；不足时使用 1.0 中性值并显式标记
`cost_baseline_ready=false`。

`model_profile.py` 将真实调用、高频可用性探针和 `traffic_type=capability_probe`
的标准任务结果按模型和日期对齐。高频探针判断模型服务状态，标准任务判断
能力、质量和一致性。真实调用异常而两类主动数据正常时，优先判断为平台、
网络或业务流量问题；真实调用与主动数据同步下降时，提高模型侧异常判断并
建议切换。融合阈值在 `Diagnosis Policy` 中维护，画像权重在 `Scoring Policy`
中维护。主路由除了综合就绪度达标，还必须满足稳定性和可信度下限，避免单项
短板被其他高分完全补偿。

`model_health_risk.py` 将性能下降、成功率异常和成本异常分别转换为 0—100
风险分，再按 `Scoring Policy` 的 `risk` 权重生成统计风险。最终风险取统计
风险、最高单项风险保护分和融合诊断下限三者最大值，防止模型拥塞或供应商
故障被平均权重稀释。所有阈值均在 `Risk Policy` 中维护；输出同时解释异常、
可能原因、是否需要切换、合格替代模型和推荐动作。替代模型必须跨供应商、
当日健康、画像角色合格，并按路由就绪度排序。

如果只修改了指标字典中的复合规则或条件，只需重新执行最后三条中的前两条，然后刷新看板：

```bash
python src/composite_rule_engine.py
python src/model_benchmark.py
python src/fusion_rule_engine.py
streamlit run dashboard/app.py
```

## 主要文件

- `requirements.txt`：云端和 pip 使用的锁定依赖，是唯一的 Streamlit Cloud 依赖入口。
- `environment.local.yml`：本地 Conda 环境配置，不会被 Streamlit Cloud 自动识别。
- `scripts/check_deployment.py`：部署资产和依赖入口一致性检查。
- `scripts/rebuild_demo.sh`：重建模拟数据、检测结果、模型评分与画像的完整流水线。
- `.github/workflows/ci.yml`：公开仓库持续集成检查。
- `docs/ai_monitoring_metric_dictionary.xlsx`：指标、单条件规则、复合规则和规则条件。
- `data/synthetic_logs_v2.csv`：30 天模拟调用日志。
- `data/ground_truth.csv`：与日志特征分离的 6 个异常事件标注。
- `outputs/features/`：平台、客户、模型、供应商和 API Key 特征表。
- `outputs/composite_alerts.csv`：复合规则告警与阈值证据。
- `outputs/benchmark/model_benchmark_results.csv`：四种检测方法的对比结果。
- `outputs/fusion_alerts.csv`：默认均衡策略生成的分层融合告警。
- `outputs/benchmark/fusion_strategy_results.csv`：高精度、均衡和高召回三套融合策略对比。
- `data/probe_runs.csv`：主动拨测运行记录。
- `outputs/probe_hourly_metrics.csv`：拨测可用率与延迟的小时聚合结果。
- `outputs/probe_alerts.csv`：连续失败、关联故障与恢复事件。
- `data/capability_probe_runs.csv`：标准任务 × 模型的对称能力校准明细。
- `outputs/model_capability_scores.csv`：模型在四个能力维度上的质量、稳定性和性能统计。
- `src/capability_calibration.py`：能力任务加载、规则评测、对称拨测与维度评分运行器。
- `outputs/model_operating_scores.csv`：真实调用侧的模型日级运营指标和配置化评分。
- `outputs/model_operating_snapshot.csv`：每个模型最新健康指数、等级和排行。
- `src/model_operations.py`：真实调用聚合、成本趋势基线、运营评分和健康排行流水线。
- `outputs/model_fusion_diagnosis.csv`：真实表现、标准拨测表现、差异、原因和建议动作。
- `outputs/model_capability_profiles.csv`：能力、稳定性、性能、可信度和路由就绪度画像。
- `src/model_profile.py`：控制变量融合诊断、画像评分和路由角色建议。
- `outputs/model_health_risks.csv`：模型日级性能、成功率、成本风险与融合风险评分。
- `outputs/model_diagnostic_evidence.csv`：异常解释、原因、切换建议、替代模型和证据摘要。
- `src/model_health_risk.py`：健康风险识别、单项保护、诊断解释和替代路由决策引擎。
- `dashboard/app.py`：Streamlit 实验看板。

## 决策看板模块

看板按决策链路组织为六个模块：运营总览输出模型健康排行；性能诊断展示
P50/P95/P99、延迟波动和性能评分；成本分析展示单请求成本、Token 成本、
成本趋势和成本性能评分；能力校准融合标准任务与真实调用差异并
输出路由画像；智能检测输出性能、成功率和成本健康风险；诊断解释中心说明
异常、可能原因、模型切换判断和推荐动作。原有异常检测算法对比、可用性拨测、
融合告警和规则配置保留在相应模块的实验或原始证据区域中。

## 模型运营评分配置

指标字典中的 `Scoring Policy` 工作表统一维护模型运营评分的组件、方向、
权重、目标值、波动容忍上限和分级区间。`src/model_scoring.py` 负责读取并
校验配置，页面和后续离线流水线不应重复硬编码权重。

当前评分族包括延迟、稳定性、性能、成功率、成本效率、成本性能、健康指数、
健康风险和画像可信度。修改配置后可运行以下测试检查权重、边界和工作簿接入：

```bash
python -m unittest discover -s tests -v
```

`Risk Policy` 进一步维护风险基线窗口、预警/严重阈值、单项信号保护系数、
融合诊断下限、解释中心准入阈值和替代模型健康下限。历史不足时趋势信号不
参与计算，但绝对阈值和主动拨测融合诊断仍然有效。

## 如何修改规则

1. 打开指标字典的 `Composite Rules` 工作表，修改规则状态、基础风险、冷却时间等。
2. 在 `Rule Conditions` 工作表中用 `rule_id` 关联条件，修改指标、比较符、静态阈值或历史基线倍数。
3. 保存 Excel 后重新运行复合规则引擎和模型评测。

`Fusion Strategies` 和 `Severity Policy` 工作表独立控制算法投票、上下文门槛、三级告警、跨层抑制和冷却时间；原复合规则及条件不会被融合层修改。

最终等级不再直接等于规则里的 `severity`：系统使用“基础风险权重 + 超阈值幅度 + 多条件命中奖励”计算分级分数。默认分数小于 0.30 为 `info`，0.30 至 0.80 为 `warning`，0.80 及以上为 `critical`；这些边界可在 `Severity Policy` 中编辑。

`Active Probes` 和 `Probe Assertions` 工作表独立控制探针，不会修改原复合规则或融合策略。保存 Excel 后，重新运行 `probe_runner.py`（模拟）或重启 `probe_scheduler.py`（实时）即可加载新配置。

`Capability Tasks` 独立维护模型能力校准任务。每个启用模型必须覆盖相同任务集合和样本数；运行器会在落盘前校验矩阵对称性。任务输入、评测器、环境版本和预期输出版本共同构成控制变量，修改后重新运行 `capability_calibration.py` 即可刷新能力数据。

当前支持的比较符是 `gt`、`gte`、`lt`、`lte`、`eq`、`neq`；复合逻辑支持 `all` 和 `any`；历史基线支持类似 `previous_24_hours_median` 的写法。

## 首轮实验结论

- 复合规则：解释性和准确率最高，当前覆盖已配置的 4 类事件。
- STL：事件覆盖更高，但误报也明显更多。
- 滚动 MAD：实现简单、适合在线计算，是较好的统计基线。
- Isolation Forest：可以发现多指标联合偏移，但仍需更多正常数据和阈值校准。

本项目中的模拟数据和评测结果用于学习与原型验证，不代表生产环境效果。
