"""Tokenizer-aware budgeting for answer-level rebuttal experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from trajpatch.analysis.context_cost import estimate_context_tokens


class _Codec(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class TokenCounter:
    """Count and truncate text with an explicit, recorded strategy."""

    name: str
    exact: bool
    codec: _Codec | None = None

    def count(self, text: str) -> int:
        value = str(text or "")
        if self.codec is not None:
            return len(self.codec.encode(value))
        return estimate_context_tokens(value)

    def truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        value = str(text or "")
        if max_tokens <= 0:
            return "", bool(value)
        if self.codec is not None:
            token_ids = self.codec.encode(value)
            if len(token_ids) <= max_tokens:
                return value, False
            return self.codec.decode(token_ids[:max_tokens]).strip(), True
        if self.count(value) <= max_tokens:
            return value, False
        word_limit = max(1, math.floor(max_tokens / 1.3))
        return " ".join(value.split()[:word_limit]), True


class _TiktokenCodec:
    def __init__(self, model_name: str) -> None:
        try:
            import tiktoken
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The tiktoken token counter requires the `tiktoken` package."
            ) from exc
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")
        self.name = str(getattr(self._encoding, "name", "o200k_base"))

    def encode(self, text: str) -> list[int]:
        return list(self._encoding.encode(str(text or "")))

    def decode(self, token_ids: list[int]) -> str:
        return str(self._encoding.decode(token_ids))


class _HuggingFaceCodec:
    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Hugging Face token counter requires `transformers`."
            ) from exc
        self._tokenizer: Any = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.name = str(
            getattr(self._tokenizer, "name_or_path", None) or model_name
        )

    def encode(self, text: str) -> list[int]:
        return list(
            self._tokenizer.encode(
                str(text or ""),
                add_special_tokens=False,
            )
        )

    def decode(self, token_ids: list[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        )

    def encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self._tokenizer(
            str(text or ""),
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(value) for value in list(encoded["input_ids"])]
        offsets = [
            (int(start), int(end))
            for start, end in list(encoded["offset_mapping"])
        ]
        if len(token_ids) != len(offsets):
            raise RuntimeError(
                "Tokenizer returned mismatched token and offset lengths."
            )
        return token_ids, offsets


def build_token_counter(
    kind: str,
    *,
    model_name: str,
    require_exact: bool = False,
) -> TokenCounter:
    """Resolve a counter without silently weakening a required formal protocol."""

    requested = str(kind or "auto").strip().lower().replace("_", "-")
    if requested not in {"auto", "tiktoken", "hf", "estimate"}:
        raise ValueError(
            "token_counter must be one of: auto, tiktoken, hf, estimate."
        )

    errors: list[str] = []
    candidates: list[str]
    if requested == "auto":
        lowered_model = str(model_name or "").casefold()
        candidates = (
            ["tiktoken", "hf"]
            if any(name in lowered_model for name in ["gpt-", "o1", "o3", "o4"])
            else ["hf", "tiktoken"]
        )
    else:
        candidates = [requested]

    for candidate in candidates:
        if candidate == "estimate":
            break
        try:
            codec: _Codec
            if candidate == "tiktoken":
                codec = _TiktokenCodec(model_name)
                return TokenCounter(
                    name=f"tiktoken:{codec.name}",
                    exact=True,
                    codec=codec,
                )
            codec = _HuggingFaceCodec(model_name)
            return TokenCounter(
                name=f"huggingface:{codec.name}",
                exact=True,
                codec=codec,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    if require_exact:
        detail = "; ".join(errors) or "no exact tokenizer was attempted"
        raise RuntimeError(
            f"Unable to construct an exact token counter for {model_name!r}: {detail}"
        )
    return TokenCounter(name="whitespace_x1_3_v1", exact=False)
