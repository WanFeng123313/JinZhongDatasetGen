from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from pretrain_dataset.config import Config

logger = logging.getLogger("pretrain_dataset")


class LLMError(RuntimeError):
    pass


class TruncatedError(LLMError):
    def __init__(self, partial: str) -> None:
        super().__init__("模型输出被长度截断")
        self.partial = partial


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def complete(self, messages: list[dict], max_tokens: int) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._once(messages, max_tokens)
            except TruncatedError:
                raise
            except LLMError as exc:
                last_error = exc
                logger.warning("模型请求失败 %s/%s：%s", attempt, self.config.max_retries, exc)
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 10))
        raise LLMError(str(last_error))

    def _once(self, messages: list[dict], max_tokens: int) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(str(exc)) from exc

        try:
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应缺少正文：{str(body)[:300]}") from exc
        if choice.get("finish_reason") == "length":
            raise TruncatedError(content)
        if not content.strip():
            raise LLMError("模型返回空内容")
        return content

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
