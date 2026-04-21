"""Gemini 2.5 Flash paraphraser with disk-backed cache.

Used by rules 2 (schema-paraphrase-invariance), 3 (synonym-robustness),
and 5 (refusal-consistency) to generate text variants without invoking
Gemini on the Stage 3 sweep's critical path.

Usage pattern
-------------

Run `scripts/generate_paraphrases.py` once, offline, to populate the cache
into `runs/paraphrase_cache/`. Commit the cache file alongside the code so
the sweep is fully reproducible. During the sweep, rules call
`paraphrase(..., offline=True)` which reads the cache and raises
`ParaphraseCacheMiss` if a key is missing. This guarantees the sweep
never makes a live API call.

Design invariants
-----------------

* **Deterministic cache keys.** Key = hash of (rule_id, input_text, variant_idx,
  model_name, temperature). No wall-clock, no random state, no caller
  identity. Same inputs → same key → same cached output across machines.
* **JSONL on disk.** One line per entry: {"key", "rule_id", "input", "output",
  "model", "timestamp"}. Append-only. Concurrent writes are not safe —
  paraphrase generation is a single-threaded offline step.
* **Soft fallback.** If a paraphrase-needing rule can't find a cache entry,
  it falls back to a template mutation (e.g. prepend a generic synonym
  marker). Logged as metadata so downstream analysis can exclude cache-
  miss pairs from Stage 3 figures if desired.
* **No torch dependency.** Pure I/O + Google client. Test-friendly without
  the `[models]` extra.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# -- Defaults + constants ----------------------------------------------------


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_CACHE_DIR = Path("runs/paraphrase_cache")
DEFAULT_TEMPERATURE = 0.2  # slight variety; still reproducible via seed


class ParaphraseCacheMiss(KeyError):
    """Raised when `offline=True` and the requested paraphrase isn't cached."""


# -- Cache key ---------------------------------------------------------------


def cache_key(
    *,
    rule_id: str,
    input_text: str,
    variant_idx: int = 0,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Stable SHA-256 hash of the paraphrase-determining inputs.

    `variant_idx` lets a single (rule, input) cache up to N paraphrase
    variants — rule 3 (synonym-robustness) for example needs 3 paraphrases
    of each user prompt.
    """
    payload = json.dumps(
        {
            "rule_id": rule_id,
            "input_text": input_text,
            "variant_idx": variant_idx,
            "model": model,
            "temperature": round(temperature, 6),
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# -- Disk cache --------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    rule_id: str
    input_text: str
    output: str
    variant_idx: int
    model: str
    temperature: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=d["key"],
            rule_id=d["rule_id"],
            input_text=d["input_text"],
            output=d["output"],
            variant_idx=int(d.get("variant_idx", 0)),
            model=d.get("model", DEFAULT_MODEL),
            temperature=float(d.get("temperature", DEFAULT_TEMPERATURE)),
            timestamp=float(d.get("timestamp", 0.0)),
        )


class ParaphraseCache:
    """File-backed cache for paraphrase outputs.

    One JSONL file per `rule_id` keeps the cache diff-friendly in git —
    rule 2 and rule 3 live in separate files even if they share an
    `input_text`. Concurrent readers are safe; concurrent writers are not.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # In-memory map: key -> CacheEntry. Lazy-loaded per rule_id.
        self._loaded: dict[str, dict[str, CacheEntry]] = {}

    def _path(self, rule_id: str) -> Path:
        safe = rule_id.replace("-", "_")
        return self.cache_dir / f"{safe}.jsonl"

    def _load(self, rule_id: str) -> dict[str, CacheEntry]:
        if rule_id in self._loaded:
            return self._loaded[rule_id]
        path = self._path(rule_id)
        entries: dict[str, CacheEntry] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry = CacheEntry.from_dict(d)
                    entries[entry.key] = entry
        self._loaded[rule_id] = entries
        return entries

    def get(
        self,
        *,
        rule_id: str,
        input_text: str,
        variant_idx: int = 0,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str | None:
        """Return the cached paraphrase or None."""
        key = cache_key(
            rule_id=rule_id,
            input_text=input_text,
            variant_idx=variant_idx,
            model=model,
            temperature=temperature,
        )
        entry = self._load(rule_id).get(key)
        return entry.output if entry is not None else None

    def put(
        self,
        *,
        rule_id: str,
        input_text: str,
        output: str,
        variant_idx: int = 0,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> CacheEntry:
        """Append an entry to disk and update the in-memory index."""
        key = cache_key(
            rule_id=rule_id,
            input_text=input_text,
            variant_idx=variant_idx,
            model=model,
            temperature=temperature,
        )
        entry = CacheEntry(
            key=key,
            rule_id=rule_id,
            input_text=input_text,
            output=output,
            variant_idx=variant_idx,
            model=model,
            temperature=temperature,
            timestamp=time.time(),
        )
        # Ensure in-memory index exists before we mutate it.
        self._load(rule_id)
        self._loaded[rule_id][key] = entry

        path = self._path(rule_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry

    def __contains__(self, key: tuple[str, str, int]) -> bool:
        """Quick `(rule_id, key_hash, variant_idx) in cache` check."""
        rule_id, key_hash, _variant = key
        return key_hash in self._load(rule_id)


# -- Paraphrase client -------------------------------------------------------


def paraphrase(
    text: str,
    *,
    rule_id: str,
    instruction: str,
    variant_idx: int = 0,
    cache: ParaphraseCache | None = None,
    offline: bool = False,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Return a paraphrased variant of `text` for `rule_id`.

    `instruction` is the system-style guidance passed to Gemini (e.g.
    "Rewrite the following tool description using different wording but
    preserving the meaning"). Each rule owns its own instruction string
    and documents the invariant being tested.

    Behaviour:
      * If `cache` has the key, return cached output (no API call).
      * Else if `offline=True`, raise `ParaphraseCacheMiss`. The Stage 3
        sweep runs in offline mode to guarantee no live API calls during
        a potentially-long sweep.
      * Else call Gemini once, store the result in `cache` (if provided),
        and return the output.
    """
    cache = cache if cache is not None else ParaphraseCache()

    cached = cache.get(
        rule_id=rule_id,
        input_text=text,
        variant_idx=variant_idx,
        model=model,
        temperature=temperature,
    )
    if cached is not None:
        return cached

    if offline:
        raise ParaphraseCacheMiss(
            f"no cached paraphrase for rule={rule_id!r} variant={variant_idx} "
            f"input={text[:80]!r}... — populate the cache via "
            f"scripts/generate_paraphrases.py before the Stage 3 sweep."
        )

    output = _call_gemini(
        text=text,
        instruction=instruction,
        model=model,
        temperature=temperature,
    )
    cache.put(
        rule_id=rule_id,
        input_text=text,
        output=output,
        variant_idx=variant_idx,
        model=model,
        temperature=temperature,
    )
    return output


def _call_gemini(
    *,
    text: str,
    instruction: str,
    model: str,
    temperature: float,
) -> str:
    """Single Gemini call — isolated so tests can monkeypatch it.

    Uses `google-generativeai` (installed via the `[gemini]` extra). Keeps
    the import lazy so `paraphrase.py` stays importable without the
    dependency — critical for the `agentmorph` package's CI-without-extras
    design.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "`google-generativeai` not installed. Run "
            "`pip install agentmorph[gemini]` to enable live paraphrasing. "
            "Or populate the paraphrase cache offline and re-run with "
            "`offline=True`."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from https://aistudio.google.com/app/apikey, then "
            "`export GEMINI_API_KEY=...` (or .env + python-dotenv)."
        )

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(
        model,
        system_instruction=instruction,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 2048,
        },
    )
    # One-shot prompt with the text to paraphrase as the only turn.
    response = gen_model.generate_content(text)
    out = getattr(response, "text", None)
    if not out:
        # Older/newer client variants expose `parts`.
        parts = getattr(response, "parts", None) or []
        out = "\n".join(str(p) for p in parts) if parts else ""
    return out.strip()
