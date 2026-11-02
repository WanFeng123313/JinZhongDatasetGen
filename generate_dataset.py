"""用本地 vLLM 把论文 Markdown 整理成预训练数据。

每篇 Markdown 生成一个 JSON 数组，元素为：

    {
      "paper_title": "...",
      "text": "...",
      "metadata": {
        "domain": "...",
        "subdomain": "...",
        "language": "中文或 English",
        "section_path": ["章节", "小节"]
      }
    }

全部 JSON 再按行汇总为一个 JSONL。已存在的 JSON 会跳过，可以中断后继续。

示例：

    python generate_dataset.py
    python generate_dataset.py --limit 20 --concurrency 4
    python generate_dataset.py --stage merge
    python generate_dataset.py --read-pdf-pages --match Z2-10105388
    python generate_dataset.py --input-dir "xxx" --output-dir "xxx"
    python generate_dataset.py --input-dir "xxx" --output-dir "xxx" --read-pdf-pages
"""

from pretrain_dataset.cli import main

if __name__ == "__main__":
    main()
