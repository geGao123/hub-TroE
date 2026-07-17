# Week08 — 中文句对匹配 (BI / Cross)

数据：`../data/{afqmc,bq_corpus,lcqmc}/{train,validation,test}.jsonl`，每行
`{sentence1, sentence2, label}`。

## 文件清单

| 文件                  | 作用                                                       |
| --------------------- | ---------------------------------------------------------- |
| `config.py`           | 路径常量 + `get_device()` (CUDA > MPS > CPU) + logging      |
| `utils_plot.py`       | loss 曲线 / 方法对比 / badcase 分布 图表                   |
| `explore_data.py`     | 行数、label 分布、句长、重复对、随机样例                   |
| `dataset.py`          | `PairDataset`（BI/Cross）、`TripletDataset`、collate       |
| `model.py`            | `BiEncoder` / `CrossEncoder` + `BCE` / `CosEnt` / `Triplet` |
| `train.py`            | 主训练入口；支持 `--mode {bi,cross}` `--loss {pair,...}`  |
| `evaluate.py`         | Acc / P / R / F1 / AUC + classification_report             |
| `analyze_badcases.py` | 错例 jsonl + 长度/Jaccard 分桶错误率                       |
| `compare_methods.py`  | 横向对比 BI/Cross，输出 CSV + PNG 柱状图                  |
| `progress.py`         | 纯 stdlib 的单行训练进度条（无 tqdm 依赖）                 |

## 路径与设备

所有路径都在 `config.py` 里集中配置，并支持环境变量覆盖：

| 常量                  | 默认                            | 环境变量                |
| --------------------- | ------------------------------- | ----------------------- |
| `PROJECT_ROOT`        | `src/` 的父目录 (即 `week08/`) | `TROE_PROJECT_ROOT`     |
| `DATA_ROOT`           | `<project>/data`                | `TROE_DATA_ROOT`        |
| `RUNS_DIR`            | `<project>/runs`                | `TROE_RUNS_DIR`         |
| `LOGS_DIR`            | `<project>/logs`                | `TROE_LOGS_DIR`         |
| `PLOTS_DIR`           | `<project>/plots`               | `TROE_PLOTS_DIR`        |
| `BASELINE_MODEL_DIR`  | `<project>/baseline_models`     | `TROE_BASELINE_MODEL_DIR` |
| `PRETRAIN_DIR`        | `<project>/pretrain_models`     | `TROE_PRETRAIN_DIR` (legacy: `PRETRAIN_MODELS_ROOT`) |

## 预训练模型路径解析

`config.resolve_model_path(name, *, roots=...)` 把 *根目录* 和 *模型名*
两个参数组合起来定位 checkpoint。查找顺序：

1. 如果 `name` 本身就是一个已存在的路径 → 直接返回；
2. 在 `roots` 列表里逐个查 `<root>/<name>`，第一个命中的就用；
3. 都没命中 → 原样返回，让 HuggingFace Hub 去下载。

**默认 roots**（无 `--pretrain_dir` 时）：
```
[<PRETRAIN_DIR>, <BASELINE_MODEL_DIR>]
```
也就是说不传任何参数时，会先去 `<project>/pretrain_models/<name>/` 找，找不到再去
`<project>/baseline_models/<name>/`，再找不到交给 Hub。

**CLI 覆盖**：每个脚本都支持 `--pretrain_dir /path/to/root`，传了就只在这个
单根下找（不再 fallback 到默认列表）。等价 YAML key 也支持。

```bash
# 三种用法
python train.py --model_name bert-base-chinese
    # → 查 <PRETRAIN_DIR>/bert-base-chinese, <BASELINE_MODEL_DIR>/bert-base-chinese, Hub

TROE_PRETRAIN_DIR=/opt/llm/pretrain python train.py --model_name bert-base-chinese
    # → 查 /opt/llm/pretrain/bert-base-chinese, Hub fallback

python train.py --model_name bert-base-chinese --pretrain_dir /tmp/my_models
    # → 只查 /tmp/my_models/bert-base-chinese, Hub fallback

python train.py --config config.example.yaml   # YAML 里写 pretrain_dir: /opt/...
```

`compare_methods.py` 的每个 `--mode name:path` 也会先尝试 `--pretrain_dir` 再走默认列表。

设备自动选择优先级：**CUDA > MPS > CPU**。Mac Apple Silicon 用户会拿到
`mps`，Linux 服务器会拿到 `cuda`，CI 没显卡就退回 `cpu`。想强制指定：

```bash
TROE_DEVICE=cuda python train.py ...
TROE_DEVICE=mps  python train.py ...
TROE_DEVICE=cpu  python train.py ...
```

> MPS 当前 autocast 不稳，所以 MPS 路径下自动关闭 AMP。CUDA 路径默认开启
> AMP（除非传 `--no_amp`）。

## 从 YAML 读取初始化参数

四个入口 (`train.py` / `evaluate.py` / `analyze_badcases.py` / `compare_methods.py`)
都接受 `--config <yaml>`。分层优先级：

```
CLI flag (命令行显式传)  >  YAML 文件  >  脚本硬编码默认值
```

举个例子 — `configs/afqmc_bce.yaml`：

```yaml
mode: bi
loss: pair
dataset: afqmc
model_name: bert-base-chinese
epochs: 3
batch_size: 32
lr: 2.0e-5
```

使用：

```bash
# 完全交给 YAML
python train.py --config config.example.yaml

# YAML 跑主体，CLI 单点覆盖（5 个 epoch）
python train.py --config config.example.yaml --epochs 5

# 多入口共用同一份配置（YAML 里有 split/batch_size/...）
python evaluate.py --mode bi --dataset afqmc \
    --model_name runs/afqmc/bi-pair/ckpt-3000 \
    --config config.example.yaml
```

完整字段表见 `config.example.yaml`，里面是带注释的。所有 `False` 默认值的
开关 (`--no_amp`, `--no_normalize`, `--no_cuda`, `--quick_train`) 也能用
YAML 的 `true`/`false` 来覆盖。

`compare_methods.py` 的 `--mode`（可重复的 `name:path`）也能用 YAML 列表
来填：

```yaml
mode:
  - bi-pair:runs/afqmc/bi-pair/ckpt-3000
  - cross-pair:runs/afqmc/cross-pair/ckpt-3000
```

技术细节：所有可调参数默认都用 `argparse.SUPPRESS` 声明，所以脚本能精确
区分「用户没传」和「用户传了默认值」。`config.parse_with_yaml()` 负责
合并优先级，详见 `src/config.py` 顶上的 docstring。

## 输出约定

- `runs/<dataset>/<mode>-<loss>/ckpt-<step>/`   — checkpoint
- `runs/<dataset>/<mode>-<loss>/eval.json`      — 最终 metrics
- `logs/<dataset>/<mode>-<loss>.log`            — 训练 log（同时输出到 stdout）
- `plots/<dataset>_<mode>-<loss>_loss.png`      — 训练 loss / acc 曲线
- `plots/<dataset>_compare.png`                 — 方法对比柱状图
- `plots/<dataset>_<mode>_errdist.png`          — badcase 错误分布图
- `<output>.jsonl`                              — badcase 明细

## 训练模式

| `--mode` | `--loss`        | 用途                          |
| -------- | --------------- | ----------------------------- |
| `bi`     | `pair`          | BI encoder + `[u;v;|u-v|]` BCE |
| `bi`     | `cosent`        | BI encoder + 排序式 CosEnt    |
| `bi`     | `triplet`       | BI encoder + TripletMargin    |
| `bi`     | `hybrid`        | BI + pair/triplet 交替        |
| `cross`  | `pair`          | Cross-encoder + BCE           |

### Triplet 取样规则（按 spec）

- anchor = `sentence1`，正样本 = 同 row 的 `sentence2`，label 必为 1
- 负样本 = 随机从其它 `label==0` 行的 `sentence2` 中抽选
- 每个 `__getitem__` 重新抽，所以每个 epoch 看到的负样本都不重样

## 跑起来

```bash
cd src

# 1. EDA
python explore_data.py

# 2. 训练
python train.py --mode bi --loss pair    --dataset afqmc --model_name bert-base-chinese --epochs 3
python train.py --mode bi --loss triplet --dataset afqmc --model_name bert-base-chinese --epochs 3
python train.py --mode bi --loss hybrid  --dataset afqmc --model_name bert-base-chinese --epochs 3
python train.py --mode cross --loss pair --dataset afqmc --model_name bert-base-chinese --epochs 3

# 3. 评估
python evaluate.py --mode bi --dataset afqmc \
    --model_name runs/afqmc/bi-pair/ckpt-8584

# 4. 错例分析
python analyze_badcases.py --mode bi --dataset afqmc \
    --model_name runs/afqmc/bi-pair/ckpt-8584 \
    --output runs/afqmc/bi-pair/badcases.jsonl

# 5. 横向对比
python compare_methods.py --dataset afqmc \
    --mode bi-pair:runs/afqmc/bi-pair/ckpt-8584 \
    --mode cross-pair:runs/afqmc/cross-pair/ckpt-8584 \
    --mode bi-triplet:runs/afqmc/bi-triplet/ckpt-2644 \
    --output_csv compare.csv
```

无 GPU / 想 smoke 试一下：把 `--model_name` 换成任意小模型（如
`prajjwal1/bert-tiny`），并加 `--max_train_steps 30 --no_amp`。

## 进度条

训练时 stdout 是 TTY 时，`train.py` 会自动渲染一行式进度条：

```
[████████░░░░░░░░░░░░░░] 41% step 1234/3000  loss=0.234  acc=0.85  lr=1.2e-05  3.10it/s  eta=04:32
```

每 `logging_steps`（默认 50）边界会临时打断进度条，打一行完整 step log
（`loss / lr / elapsed / acc / d_pos …`），下一行进度条自动续上。训练结束
后干净退出，无 ANSI 残留。

关闭方法（任选其一）：

```bash
python train.py ... --no_progress_bar           # CLI
# 或 YAML
no_progress_bar: true
# 或把 stdout 重定向到文件（自动检测非 TTY）
python train.py ... > run.log
```

实现细节：`src/progress.py` 是 ~150 行的 stdlib 类（`█` / `░` 字符画 30 列
bar + deque 平滑 loss / acc + ETA），无 tqdm / rich 依赖，log 文件不会沾到
ANSI 转义码。