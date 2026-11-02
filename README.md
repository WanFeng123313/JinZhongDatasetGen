# 金钟预训练数据集生成

把知网论文 Markdown 整理成预训练样本。每篇 Markdown 生成一个 JSON 数组，全部 JSON 再汇总成一个 JSONL。正文按本地标题切分；模型只负责学科分类，以及改写缺标点的段落。表格和公式保持原文。

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

## 环境

生成脚本使用 Python 3.11 及以上，并安装 OpenAI SDK。本机对应 conda 环境 `openai_env`：

```bash
conda activate openai_env
pip install openai
```

模型服务是 OpenAI 兼容接口，由 vLLM 提供。调用前可用仓库里的 `test_vllm.py` 检查服务是否可用：

```bash
python test_vllm.py --base-url http://192.168.141.11:8002/v1 --model qwen3.8-27b
```

脚本会列出服务上的模型名，并要求模型只返回 `{"ok": true}`。`--model` 必须和服务里实际加载的名字一致。

开启 `--read-pdf-pages` 时，还需要系统里的 `pdftoppm`（Poppler）。Ubuntu 上可以安装：

```bash
sudo apt-get install poppler-utils
```

该开关把 PDF 页渲成图片后交给当前对话模型，不需要单独的 OCR 环境。

## 使用

在仓库根目录执行。默认读取 `/mnt/sdc/jinzhong_MD_local_final`，结果写到 `dataset_output/`。目录名里含 `_logs` 的 Markdown 会跳过。

```bash
python generate_dataset.py
```

常用例子：

```bash
# 先看会切成多少段，不调用模型
python generate_dataset.py --dry-run --limit 20

# 小批量试跑
python generate_dataset.py --limit 20 --concurrency 4 --output-dir dataset_output_trial

# 只重做某一篇
python generate_dataset.py --match Z2-10105388 --limit 1 --overwrite --stage generate --output-dir /tmp/jz_one

# 转页稿用页面图重读版面
python generate_dataset.py --read-pdf-pages --match Z2-10105388 --limit 1 --overwrite --stage generate

# 已有 JSON 时，只重新汇总 JSONL
python generate_dataset.py --stage merge --output-dir dataset_output

# 本地规则自检，不访问模型
python generate_dataset.py --self-test
```

已生成的 JSON 会跳过，中断后直接再跑即可继续。要覆盖旧结果，加上 `--overwrite`。质量过滤后变成空数组的文件同样会被跳过，需要重做时也要加 `--overwrite`。

跑完后日志给出总用时、并发数，以及平均每篇耗时。平均耗时按成功和失败的篇数计算，跳过的不计入。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input-dir` | `/mnt/sdc/jinzhong_MD_local_final` | 论文 Markdown 根目录 |
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

## 输出

```text
dataset_output/
├── json/信息科技/科技传播/Z2-10105388/hybrid_auto/Z2-10105388.json
├── pretrain_qwen3.8_27b.jsonl
├── failed.jsonl
└── run.log
```

`json/` 的相对路径与输入 Markdown 一致，后缀改为 `.json`。JSONL 中每一行是一条记录。失败篇目记在 `failed.jsonl`。开启 `--read-pdf-pages` 后，页面图缓存在 `output-dir/page_cache/<论文编号>/`。

输入 Markdown 不超过 20000 篇时，汇总为一个 `--jsonl-name` 指定的文件。超过 20000 篇时，按输入顺序每 20000 篇写成一个文件，文件名在扩展名前加序号，例如 `pretrain_qwen3.8_27b_00001.jsonl`、`pretrain_qwen3.8_27b_00002.jsonl`。最后一份可以少于 20000 篇。某一篇还没有 JSON 时，它仍占一个名额，后面的论文不会因此提前进入上一份文件。

## 处理说明

同一文件里出现两篇题录（「文章编号」出现两次，并且有第二个一级标题）时，保留第一篇，去掉后面那篇，再生成 JSON。

转页标记默认只在文本里剔除串入的其他文章并接上断句。加上 `--read-pdf-pages` 后，才会按 Markdown 的相对路径到 `--pdf-dir` 找 PDF，把前 12 页渲成图片交给模型。模型不可用或没有对上论文题目时，仍使用原来的 Markdown。
