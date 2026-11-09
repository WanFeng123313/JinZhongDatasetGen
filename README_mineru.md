# MinerU 多 GPU 本地批处理（v1.5.4）

将目录内 PDF 递归转为 Markdown。每张 GPU 独立跑本地 MinerU（无 Router / 无远程 API），支持断点续跑、进度条、**索引分片多机并行**。控制台日志带时间戳；索引模式默认不做逐文件 NAS `stat`，启动更快。**中间临时目录默认在本地盘**（`--temp-dir`，勿用 NAS），避免 CIFS `Input/output error`。多机共用 `--output` 时，批处理日志写在 `_logs/<run-id>/`（默认取分片文件名）。

- 脚本：`mineru_local_multi_gpu.py`
- 索引：`build_pdf_index.py`（扫 PDF → 全量索引 + N 分片；供 MinerU / 后续预训练共用）
- 清理：`clean_md_output.py`（可选；清理历史整包，保留 md/images/`.success`）
- 本机对齐版本：`mineru==3.4.5`（conda 环境约 8.4G；参考依赖 `torch==2.11.0` / `vllm==0.20.2`）

**v1.5.4**：跳过逻辑以**当前 `--output` 下的 `.success`** 为准；仅 TSV `ocr_status=ok` 但磁盘无 `.success` 会重做（`stale_tsv_ok_no_success`），避免换 OUT 后空跳过。

**v1.5.3**：默认 `--keep-output md-images`，落盘只留 `*.md` + `images/`；需要整包时加 `--keep-output all`。

**v1.5.2**：同卡多实例默认 `--engine-init-lock auto`，冷启 `EngineCore` 时用 flock 互斥，避免显存 profile 互踩（`Error in memory profiling` / `Engine core initialization failed`）；稳态推理仍可并行。

**v1.5.1**：喂 mineru 前跳过坏/空 PDF（`bad_pdf.txt`），避免 PdfiumError 连坐整 chunk。

**v1.5.0**：`--order small-first` + `--max-bytes`；成功后回写 TSV `ocr_status` / `ocr_finished_at`（同 shard 单写者）；`.success` 可回填 TSV。

**v1.4.0**：`--mineru-chunk-size` 隔离 EngineCore 连坐；小文件优先 / 大文件延后。

## 环境要求

- Python **3.10–3.13**（本机为 **3.12.14**）
- NVIDIA GPU + CUDA（本机 torch 为 `2.11.0+cu130`）
- 磁盘：环境约 **9G+**，另加输入 PDF / 输出 Markdown 空间

脚本与 MinerU **共用同一个环境**即可。

## 安装

清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 方式：conda

```bash
conda create -n mineru python=3.12 -y
conda activate mineru
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install "mineru[all]==3.4.5" -i https://pypi.tuna.tsinghua.edu.cn/simple
mineru --version   # 应为 3.4.5
```

首次使用如需下载模型：
```bash
mineru-models-download
```

## 索引分片（推荐：多机 / 百万级）

```bash
# 扫 PDF → 全量索引 + N 分片（默认 --balance bytes，片内 --shard-order small-first）
python build_pdf_index.py \
  --input /NAS_TYSK/02Data/大模型三水论文/20261102提供 \
  --output /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1104 \
  --shards 5

# 已有 all_pdfs.tsv：不必重扫，按体积重切 + 片内小→大
python build_pdf_index.py \
  --from-index /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1103/all_pdfs.tsv \
  --output /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1104 \
  --shards 5 \
  --balance bytes \
  --shard-order small-first
```

生成：`all_pdfs.tsv`、`shard_00.tsv`…、`shards_manifest.tsv`。  
索引列：`rel_path bytes mtime_ns batch lang doc_id`（相对建索引时的论文根目录）。  
`--balance bytes`（默认）按文件大小贪心均分；`--shard-order small-first`（默认）保证每片 FIFO 先吃小文件。  
`bytes` 均分时分配算法内部是大文件优先入桶，**若不做片内重排，队首会全是巨本**——这是 1103 看起来变慢的主因之一。

**索引只需在一台机器上建好**，拷到共享盘或各机复用即可。  
各机跑任务时：`--input` 填**本机上的论文根目录**（同一套数据，挂载路径可以不同），`--file-list` 指向分到的那份 `shard_XX.tsv`。

```bash
# 可用 1103 或 1104；1103 队首虽是巨本，脚本仍会先做 ≤10MiB
# 成功后会改写该 shard TSV（追加 ocr_status / ocr_finished_at），每机独占自己的 shard
nohup python -u mineru_local_multi_gpu.py \
  --input /NAS_TYSK/02Data/大模型三水论文/20261102提供 \
  --file-list /NAS_TYSK/02Data/大模型三水论文/20261102提供/pdf_index_1103/shard_01.tsv \
  --output /NAS_TYSK/02Data/大模型三水论文/sanshui_MD_1104 \
  --gpus 1 \
  --batch-size 1000 \
  --mineru-chunk-size 1000 \
  --max-bytes 0 \
  --order large-first \
  --gpu-memory-utilization 0.2 \
  > /NAS_TYSK/02Data/大模型三水论文/run_shard01.log 2>&1 &

tail -f /NAS_TYSK/02Data/大模型三水论文/run_shard01.log
```

多机共用同一 `--output`；各机换自己的 `shard_XX.tsv` 和 `run_shardXX.log` 即可。  
批处理日志自动落到 `{output}/_logs/shard_XX/`（由 file-list 文件名决定；也可用 `--run-id` 覆盖）。  
TSV 标记列：`ocr_status`（成功为 `ok`）、`ocr_finished_at`；跳过以当前 OUT 的 `.success` 为准，TSV 只作账本/回填。

## 用法（无索引，单机直接扫目录）

适合小数据或试跑；百万级请用上面的索引分片。

```bash
# 试跑 10 篇
python mineru_local_multi_gpu.py \
  --input /NAS_TYSK/02Data/大模型三水论文/20261102提供/中文/Zw1.1/pdf/哲学与人文科学 \
  --output /mnt/sdc/sanshui_MD_test \
  --gpus 0,1 \
  --batch-size 50 \
  --mineru-chunk-size 10 \
  --gpu-memory-utilization 0.45 \
  --limit 10

# 单机扫整个子目录（后台）
nohup python -u mineru_local_multi_gpu.py \
  --input /NAS_TYSK/02Data/大模型三水论文/20261102提供/中文/Zw1.1/pdf/哲学与人文科学 \
  --output /mnt/sdc/sanshui_MD_test \
  --gpus 0,1 \
  --batch-size 200 \
  --mineru-chunk-size 25 \
  --gpu-memory-utilization 0.45 \
  > /mnt/sdc/sanshui_MD_test/run.log 2>&1 &

tail -f /mnt/sdc/sanshui_MD_test/run.log
```

### 停止

```bash
# 若正在 tail 日志，先 Ctrl+C 退出查看

# 停掉批处理脚本及子进程 mineru（含其拉起的临时 fast_api / EngineCore）
pkill -f mineru_local_multi_gpu.py; pkill -f 'mineru -p'; pkill -f mineru.cli.fast_api
# 若仍有孤儿占卡：
# pgrep -af 'mineru.cli.fast_api|VLLM::EngineCore'

# 或按 nohup 打印的 PID：
# kill <PID>
# 几秒后仍在则：kill -9 <PID> && pkill -f 'mineru -p'

# 确认已停（无输出即干净）
pgrep -af 'mineru_local_multi_gpu|mineru -p'
```

当前 `--output` 下已有 `.success` 的结果会保留，重跑会自动跳过；仅 TSV 已 `ok` 但 OUT 无 `.success` 不会跳过。

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input` | 见脚本 | PDF 根 / 索引 rel_path 解析根 |
| `--file-list` | 无 | 索引或分片 TSV；指定后只处理列表内文件 |
| `--output` | 见脚本 | Markdown 输出根目录 |
| `--run-id` | file-list stem / 主机名 | `_logs/<run-id>/` 子目录；多机隔离日志 |
| `--temp-dir` | `/tmp/mineru_local_batches` | 本地临时目录（MinerU 中间结果）；勿指向 NAS |
| `--gpus` | `1,2` | GPU ID，逗号分隔、不可重复 |
| `--batch-size` | `200` | 调度/记账粒度（失败列表、落盘批次） |
| `--mineru-chunk-size` | `25` | **单次 mineru CLI 篇数**；隔离 EngineCore 挂掉后的连坐 |
| `--max-bytes` | `10485760` | 大于该体积的 PDF 延后到队尾；`0` 关闭（与索引片内 small-first 叠加更稳） |
| `--order` | `small-first` | `as-is` / `small-first` / `large-first`；即使用 1103 也会运行时重排 |
| `--keep-output` | `md-images` | `md-images`=只留 md+images；`all`=MinerU 整包 |
| （自动） | — | 成功后回写 file-list：`ocr_status=ok`、`ocr_finished_at`；同 shard 单写者 |
| `--limit` | `0` | `>0` 时只处理前 N 个；`0` 表示全部 |
| `--backend` | `hybrid-engine` | 传给 `mineru -b` |
| `--gpu-memory-utilization` | `0.3` | 传给 VLLM 的显存占比 `(0,1]`；**32GB 卡建议 `0.45`**（给 OCR/layout 留空），96GB 可用 `0.7`。略降可能稍慢，但比 OOM 失败重试划算 |
| `--progress-interval` | `30` | 总体进度心跳间隔（秒）；`0` 关闭 |

进度会写入 `run.log`：`[progress]` 总进度条 + ETA，`[GPU n]` batch 内 `task#` / 错误摘要，`[live]` 各卡当前 batch。

更细的 MinerU 原文日志：

```bash
tail -f {output}/_logs/<run-id>/gpu*_batch*.log
```

超大文件延后清单：`{output}/_logs/<run-id>/deferred_large.txt`。

## 输出约定

- 每个 PDF → `{output}/{相对路径}/{stem}/`，默认内含 `*.md`、`images/` 与 `.success` 标记
- 当前 OUT 已有 `.success` → **跳过**（可安全重跑）；TSV `ok` 仅作账本/回填，不足以单独跳过
- 日志目录：`{output}/_logs/<run-id>/`（默认 `shard_XX` 或主机名）
- 批日志：`gpu{N}_batchXXXXX.log`；失败列表：`failed_local.txt`（追加写入）

## 注意

1. **目录扫描模式**：同名 PDF（忽略大小写）在不同子目录会直接报错退出。**索引模式**按 `rel_path` 判重，允许不同目录同文件名。
2. 不要设置 `MINERU_API_URL`；脚本会清除该变量，强制本地引擎。
3. 请固定 `mineru==3.4.5`，勿随意升到 4.x（接口/行为可能不兼容本脚本）。
4. 多机并行时各分片 `--file-list` 必须互斥；索引可共享复用，`--input` 为本机论文根目录（能解析索引里的 `rel_path` 即可）。
5. `--temp-dir` 必须可写本地盘。把临时目录放在 NAS 上容易出现 `OSError: Input/output error`，导致整批失败。
6. 结果落到 CIFS/NAS 时脚本只拷贝文件内容（v1.3.4+），避免 `shutil.move`/`copy2` 触发 `chmod` → `Operation not permitted`。
7. **勿在未停旧任务时换分片或改参数中途重启**；先 `pkill` 干净，确认无孤儿 `fast_api`/`EngineCore` 再启 v1.4.0。
