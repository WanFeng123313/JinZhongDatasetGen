from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    base_url: str = "http://192.168.141.73:8002/v1"
    model: str = "nanbeige4.3-33b"
    api_key: str = ""
    concurrency: int = 16
    limit: int = 0
    match: str = ""
    overwrite: bool = False
    min_chars: int = 40
    single_pass_chars: int = 16000
    chunk_chars: int = 7000
    max_retries: int = 3
    parse_retries: int = 2
    timeout: float = 600.0
    temperature: float = 0.0
    max_output_tokens: int = 32768
    stage: str = "all"
    jsonl_name: str = "pretrain_qwen3.8_27b.jsonl"
    dry_run: bool = False
    pdf_dir: Path = Path("/mnt/sdc/金钟大模型文献/中文10000样品/Pdf")
    ocr_python: str = ""
    read_pdf_pages: bool = False

    @property
    def json_dir(self) -> Path:
        return self.output_dir / "json"

    @property
    def jsonl_path(self) -> Path:
        return self.output_dir / self.jsonl_name

    @property
    def failed_path(self) -> Path:
        return self.output_dir / "failed.jsonl"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run.log"
