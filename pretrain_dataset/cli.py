from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pretrain_dataset.config import Config
from pretrain_dataset.llm_client import LLMClient
from pretrain_dataset.pipeline import (
    PaperProcessor,
    Progress,
    discover_markdowns,
    merge_jsonl,
    output_json_path,
    record_failure,
    write_paper_json,
)
from pretrain_dataset.quality import self_check as quality_self_check
from pretrain_dataset.text_norm import self_check

logger = logging.getLogger("pretrain_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用本地 vLLM 将论文 Markdown 整理为预训练 JSON/JSONL。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/sdc/jinzhong_MD_local_final"),
        help="论文 Markdown 根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/xyguo/code/jinzhong_pretrain_dataset/dataset_output"),
        help="JSON 与 JSONL 输出目录",
    )
    parser.add_argument("--base-url", default="http://192.168.141.11:8002/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-key", default="empty", help="可选，默认读取 OPENAI_API_KEY")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="同时发给模型的请求数。显存充足时可试 24 或 32",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇，0 表示全部")
    parser.add_argument("--match", default="", help="只处理路径中包含该字符串的文件")
    parser.add_argument("--overwrite", action="store_true", help="重做已经生成的 JSON")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--single-pass-chars", type=int, default=16000)
    parser.add_argument("--chunk-chars", type=int, default=7000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "merge"),
        default="all",
        help="all：生成 JSON 后汇总；generate：只生成 JSON；merge：只汇总已有 JSON",
    )
    parser.add_argument("--jsonl-name", default="pretrain_qwen3.8_27b.jsonl")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("/mnt/sdc/金钟大模型文献/中文10000样品/Pdf"),
        help="与 Markdown 对应的原 PDF 根目录，转页时用来重解析",
    )
    parser.add_argument(
        "--ocr-python",
        default="",
        help="本地 OCR 环境的 Python。默认使用 ocr_env",
    )
    parser.add_argument(
        "--read-pdf-pages",
        action="store_true",
        help="遇到上接第/下转第时，把原 PDF 页面图交给模型读版面后再生成 JSON。默认关闭",
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计切分，不调用模型")
    parser.add_argument("--self-test", action="store_true", help="运行本地文本规则自检")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
        concurrency=max(1, args.concurrency),
        limit=max(0, args.limit),
        match=args.match,
        overwrite=args.overwrite,
        min_chars=args.min_chars,
        single_pass_chars=args.single_pass_chars,
        chunk_chars=args.chunk_chars,
        timeout=args.timeout,
        stage=args.stage,
        jsonl_name=args.jsonl_name,
        dry_run=args.dry_run,
        pdf_dir=args.pdf_dir,
        ocr_python=args.ocr_python,
        read_pdf_pages=args.read_pdf_pages,
    )


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    return f"{minutes}分{secs}秒"


def generate(config: Config) -> Progress:
    files = discover_markdowns(config)
    if not files:
        raise SystemExit(f"没有找到 Markdown：{config.input_dir}")
    logger.info(
        "待处理 %s 篇，并发 %s，模型 %s @ %s，PDF 页面图重读 %s",
        len(files),
        config.concurrency,
        config.model,
        config.base_url,
        "开启" if config.read_pdf_pages else "关闭",
    )
    processor = PaperProcessor(config, LLMClient(config))
    if config.dry_run:
        for path in files:
            logger.info(processor.describe(path))
        return Progress(0)

    progress = Progress(len(files))
    fail_lock = threading.Lock()

    def worker(path: Path) -> None:
        source = str(path.relative_to(config.input_dir))
        destination = output_json_path(config, path)
        if destination.exists() and not config.overwrite:
            progress.update("skip", source)
            return
        try:
            records = processor.process_file(path)
        except Exception as exc:
            record_failure(config, path, str(exc), fail_lock)
            logger.exception("处理失败：%s (%s)", source, exc)
            progress.update("fail", source)
            return
        write_paper_json(destination, records)
        progress.update("ok", source, len(records))

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [executor.submit(worker, path) for path in files]
        for future in as_completed(futures):
            future.result()
    logger.info(
        "生成结束：成功 %s，跳过 %s，失败 %s，样本 %s",
        progress.ok,
        progress.skip,
        progress.fail,
        progress.records,
    )
    return progress


def log_run_summary(config: Config, progress: Progress | None, elapsed: float) -> None:
    processed = 0 if progress is None else progress.ok + progress.fail
    average = format_duration(elapsed / processed) if processed else "-"
    logger.info(
        "总用时 %s，并发数 %s，平均每篇耗时 %s（按成功和失败共 %s 篇计算，不含跳过）",
        format_duration(elapsed),
        config.concurrency,
        average,
        processed,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_check()
        quality_self_check()
        print("self-test ok")
        return
    config = config_from_args(args)
    if not config.input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{config.input_dir}")
    setup_logging(config.log_path)
    started = time.perf_counter()
    progress = None
    if config.stage in {"all", "generate"}:
        progress = generate(config)
    if config.stage in {"all", "merge"} and not config.dry_run:
        merge_jsonl(config)
    if progress is not None and not config.dry_run:
        log_run_summary(config, progress, time.perf_counter() - started)
