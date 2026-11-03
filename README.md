# 金钟预训练数据集

流水线分两步。第一步把 PDF 转成 Markdown，第二步把 Markdown 整理成预训练样本。

| 顺序 | 脚本 | 作用 |
| --- | --- | --- |
| 0 | `build_pdf_index.py` | 扫 PDF → 全量索引 + N 分片（多机分配） |
| 1 | `01_mineru_local_multi_gpu.py` | PDF 递归转为 Markdown（可按索引分片） |
| 2 | `02_generate_dataset.py` | Markdown 整理为 JSON，再汇总为 JSONL |

第一步的 `--output` 目录，就是第二步的 `--input-dir`。两个脚本使用不同的 conda 环境。

## 第一步：PDF 转 Markdown

`01_mineru_local_multi_gpu.py`（v1.2.0）把目录内 PDF 递归转为 Markdown。每张 GPU 独立跑本地 MinerU（无 Router / 无远程 API），支持断点续跑、进度条、**索引分片多机并行**。

索引工具：`build_pdf_index.py`（扫 PDF → 全量索引 + N 分片；供 MinerU / 后续预训练共用）。

本机对齐版本：`mineru==3.4.5`（conda 环境约 8.4G；参考依赖 `torch==2.11.0` / `vllm==0.20.2`）。

### 环境

- Python **3.10–3.13**（本机为 **3.12.14**）
- NVIDIA GPU + CUDA（本机 torch 为 `2.11.0+cu130`）
- 磁盘：环境约 **9G+**，另加输入 PDF / 输出 Markdown 空间

脚本与 MinerU 共用同一个环境。清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

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

请固定 `mineru==3.4.5`。不要设置 `MINERU_API_URL`；脚本会清除该变量，强制本地引擎。

### 索引分片（推荐：多机 / 百万级）

在仓库根目录执行：

```bash
# 扫 PDF → 自动建输出目录 → all_pdfs.tsv + shard_00/01/02.tsv
python build_pdf_index.py \
  --input /path/to/pdf_root \
  --output /path/to/pdf_index \
  --shards 3
```

生成：`all_pdfs.tsv`、`shard_00.tsv`…`shard_02.tsv`、`shards_manifest.tsv`。  
索引列：`rel_path bytes mtime_ns batch lang doc_id`（相对建索引时的论文根目录）。

**索引只需在一台机器上建好**，拷到共享盘或各机复用即可。  
各机跑任务时：`--input` 填**本机上的论文根目录**（同一套数据，挂载路径可以不同），`--file-list` 指向分到的那份 `shard_XX.tsv`。

```bash
# 机0 / 机1 / 机2：只改 shard_00 → shard_01 → shard_02
nohup python -u 01_mineru_local_multi_gpu.py \
  --input /path/to/pdf_root \
  --file-list /path/to/pdf_index/shard_00.tsv \
  --output /path/to/md_out \
  --gpus 0,1 \
  --batch-size 500 \
  > /path/to/md_out/run_shard00.log 2>&1 &

tail -f /path/to/md_out/run_shard00.log
```

三机可共用同一 `--output`；各机换自己的 `shard_XX.tsv` 和 `run_shardXX.log` 即可。

### 用法（无索引，单机直接扫目录）

适合小数据或试跑；百万级请用上面的索引分片。在仓库根目录、`mineru` 环境中执行：

```bash
# 前台试跑（限制数量）
python 01_mineru_local_multi_gpu.py \
  --input /path/to/pdf_root \
  --output /path/to/md_out \
  --gpus 0,1 \
  --batch-size 100 \
  --limit 10

# 后台大批量
nohup python -u 01_mineru_local_multi_gpu.py \
  --input /path/to/pdf_root \
  --output /path/to/md_out \
  --gpus 0,1 \
  --batch-size 1000 \
  > /path/to/run.log 2>&1 &

tail -f /path/to/run.log
```

停止：

```bash
# 若正在 tail 日志，先 Ctrl+C 退出查看

# 停掉批处理脚本及子进程 mineru
pkill -f 01_mineru_local_multi_gpu.py
pkill -f 'mineru -p'

# 或按 nohup 打印的 PID：
# kill <PID>
# 几秒后仍在则：kill -9 <PID> && pkill -f 'mineru -p'

# 确认已停（无输出即干净）
pgrep -af '01_mineru_local_multi_gpu|mineru -p'
```

已写出 `.success` 的结果会保留，重跑会自动跳过。

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--input` | 见脚本 | PDF 根 / 索引 `rel_path` 解析根 |
| `--file-list` | 无 | 索引或分片 TSV；指定后只处理列表内文件 |
| `--output` | 见脚本 | Markdown 输出根目录 |
| `--gpus` | `0,1,2` | GPU ID，逗号分隔、不可重复 |
| `--batch-size` | `1000` | 每批 PDF 数 |
| `--limit` | `0` | `>0` 时只处理前 N 个；`0` 表示全部 |
| `--backend` | `hybrid-engine` | 传给 `mineru -b` |
| `--gpu-memory-utilization` | `0.7` | GPU 显存占用比例 `(0,1]` |
| `--progress-interval` | `30` | 总体进度心跳间隔（秒）；`0` 关闭 |

进度会写入 `run.log`：`[progress]` 总进度条 + ETA，`[GPU n]` batch 内 `task#` / 错误摘要，`[live]` 各卡当前 batch。更细的 MinerU 原文日志：

```bash
tail -f {output}/_logs/gpu*_batch*.log
```

### 输出

- 每个 PDF → `{output}/{相对路径}/{stem}/`，内含 `.md` 与 `.success` 标记
- 已有 `.success` 且存在非空 `.md` → 跳过（可安全重跑）
- 日志：`{output}/_logs/gpu{N}_batchXXXXX.log`
- 失败列表：`{output}/_logs/failed_local.txt`（追加写入）

### 注意

1. **目录扫描模式**：同名 PDF（忽略大小写）在不同子目录会直接报错退出。**索引模式**按 `rel_path` 判重，允许不同目录同文件名。
2. 不要设置 `MINERU_API_URL`；脚本会清除该变量，强制本地引擎。
3. 请固定 `mineru==3.4.5`，勿随意升到 4.x（接口/行为可能不兼容本脚本）。
4. 多机并行时各分片 `--file-list` 必须互斥；索引可共享复用，`--input` 为本机论文根目录（能解析索引里的 `rel_path` 即可）。

## 第二步：Markdown 生成预训练数据

`02_generate_dataset.py` 把知网论文 Markdown 整理成预训练样本。每篇 Markdown 生成一个 JSON 数组，全部 JSON 再汇总成一个 JSONL。正文按本地标题切分；模型只负责学科分类，以及改写缺标点的段落。表格和公式保持原文。

一条记录的格式：

```json
{
  "paper_title": "论文题目",
  "text": "该章节正文，表格为 Markdown 管道表，公式为 LaTeX",
  "metadata": {
    "domain": "水文地质",
    "subdomain": "矿坑涌水预测",
    "language": "中文",
    "section_path": ["矿区水文地质条件", "矿床充水因素"]
  }
}
```

中文论文的 `language` 为 `中文`，`domain` 和 `subdomain` 用简短中文。英文论文的 `language` 为 `English`，学科名用简短英文。

### 环境

生成脚本使用 Python 3.11 及以上，并安装 OpenAI SDK。本机对应 conda 环境 `openai_env`：

```bash
conda activate openai_env
pip install openai
```

开启 `--read-pdf-pages` 时，还需要系统里的 `pdftoppm`（Poppler）。Ubuntu 上可以安装：

```bash
sudo apt-get install poppler-utils
```

该开关把 PDF 页渲成图片后交给当前对话模型，不需要单独的 OCR 环境。

### 用法

在仓库根目录、`openai_env` 环境中执行。`--input-dir` 指向第一步的 Markdown 输出。默认读取 `/mnt/sdc/jinzhong_MD_local_final`，结果写到 `dataset_output/`。目录名里含 `_logs` 的 Markdown 会跳过。

```bash
python 02_generate_dataset.py --input-dir "md目录" --output-dir "输出目录"
```

常用例子：

```bash
# 先看会切成多少段，不调用模型
python 02_generate_dataset.py --dry-run --limit 20

# 小批量试跑
python 02_generate_dataset.py --limit 20 --concurrency 4 --output-dir dataset_output_trial

# 只重做某一篇
python 02_generate_dataset.py --match Z2-10105388 --limit 1 --overwrite --stage generate --output-dir /tmp/jz_one

# 转页稿用页面图重读版面
python 02_generate_dataset.py --read-pdf-pages --match Z2-10105388 --limit 1 --overwrite --stage generate

# 已有 JSON 时，只重新汇总 JSONL
python 02_generate_dataset.py --stage merge --output-dir dataset_output

# 本地规则自检，不访问模型
python 02_generate_dataset.py --self-test
```

已生成的 JSON 会跳过，中断后直接再跑即可继续。要覆盖旧结果，加上 `--overwrite`。质量过滤后变成空数组的文件同样会被跳过，需要重做时也要加 `--overwrite`。

跑完后日志给出总用时、并发数，以及平均每篇耗时。平均耗时按成功和失败的篇数计算，跳过的不计入。

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-dir` | `/mnt/sdc/jinzhong_MD_local_final` | 论文 Markdown 根目录，即第一步的输出目录 |
| `--output-dir` | `dataset_output` | JSON、JSONL 和日志的输出目录 |
| `--base-url` | `http://192.168.141.11:8002/v1` | vLLM 的 OpenAI 兼容地址 |
| `--model` | `qwen3.8-27b` | 服务中的模型名 |
| `--api-key` | `empty` | 接口密钥。传入空字符串时改读环境变量 `OPENAI_API_KEY` |
| `--concurrency` | `16` | 同时发给模型的请求数 |
| `--limit` | `0` | 只处理前 N 篇，`0` 表示全部 |
| `--match` | 空 | 只处理路径中包含该字符串的文件 |
| `--overwrite` | 关闭 | 重做已经生成的 JSON |
| `--min-chars` | `40` | 短于该字数的片段不单独成条 |
| `--single-pass-chars` | `16000` | 不超过该长度的论文走单次整理 |
| `--chunk-chars` | `7000` | 超长论文按这个长度分块 |
| `--timeout` | `600` | 单次模型请求的超时时间，单位秒 |
| `--stage` | `all` | `all` 先生成再汇总；`generate` 只生成 JSON；`merge` 只汇总已有 JSON |
| `--jsonl-name` | `pretrain_qwen3.8_27b.jsonl` | 汇总文件名 |
| `--pdf-dir` | `/mnt/sdc/金钟大模型文献/中文10000样品/Pdf` | 与 Markdown 对应的原 PDF 根目录 |
| `--read-pdf-pages` | 关闭 | Markdown 含「上接第」「下转第」等转页标记时，把 PDF 页面图交给模型重读版面 |
| `--ocr-python` | 空 | 本地 OCR 用的 Python。留空时使用 `~/miniconda3/envs/ocr_env/bin/python` |
| `--dry-run` | 关闭 | 只统计切分，不调用模型 |
| `--self-test` | 关闭 | 运行本地文本规则自检 |

### 输出

```text
dataset_output/
├── json/信息科技/科技传播/Z2-10105388/hybrid_auto/Z2-10105388.json
├── pretrain_qwen3.8_27b.jsonl
├── failed.jsonl
└── run.log
```

`json/` 的相对路径与输入 Markdown 一致，后缀改为 `.json`。JSONL 中每一行是一条记录。失败篇目记在 `failed.jsonl`。开启 `--read-pdf-pages` 后，页面图缓存在 `output-dir/page_cache/<论文编号>/`。

输入 Markdown 不超过 20000 篇时，汇总为一个 `--jsonl-name` 指定的文件。超过 20000 篇时，按输入顺序每 20000 篇写成一个文件，文件名在扩展名前加序号，例如 `pretrain_qwen3.8_27b_00001.jsonl`、`pretrain_qwen3.8_27b_00002.jsonl`。最后一份可以少于 20000 篇。某一篇还没有 JSON 时，它仍占一个名额，后面的论文不会因此提前进入上一份文件。

### 处理说明

同一文件里出现两篇题录（「文章编号」出现两次，并且有第二个一级标题）时，保留第一篇，去掉后面那篇，再生成 JSON。

转页标记默认只在文本里剔除串入的其他文章并接上断句。加上 `--read-pdf-pages` 后，才会按 Markdown 的相对路径到 `--pdf-dir` 找 PDF，把前 12 页渲成图片交给模型。模型不可用或没有对上论文题目时，仍使用原来的 Markdown。
