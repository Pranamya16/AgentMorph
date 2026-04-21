"""Tests for the paraphrase cache.

Pins:
  * `cache_key` is deterministic and sensitive to every input field.
  * `ParaphraseCache.put` + `get` round-trip through disk.
  * `paraphrase(..., offline=True)` raises `ParaphraseCacheMiss` on miss.
  * `paraphrase(...)` with `offline=False` calls the gemini client exactly
    once and caches the result (verified via monkeypatched `_call_gemini`).
  * The module imports cleanly without `google-generativeai` installed —
    the import only happens inside `_call_gemini()`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentmorph import paraphrase as mod
from agentmorph.paraphrase import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    CacheEntry,
    ParaphraseCache,
    ParaphraseCacheMiss,
    cache_key,
    paraphrase,
)


# -- cache_key determinism ---------------------------------------------------


def test_cache_key_is_deterministic() -> None:
    k1 = cache_key(rule_id="synonym-robustness", input_text="find a kettle")
    k2 = cache_key(rule_id="synonym-robustness", input_text="find a kettle")
    assert k1 == k2


def test_cache_key_sensitive_to_rule_id() -> None:
    k1 = cache_key(rule_id="synonym-robustness", input_text="X")
    k2 = cache_key(rule_id="schema-paraphrase-invariance", input_text="X")
    assert k1 != k2


def test_cache_key_sensitive_to_input_text() -> None:
    k1 = cache_key(rule_id="r", input_text="a")
    k2 = cache_key(rule_id="r", input_text="b")
    assert k1 != k2


def test_cache_key_sensitive_to_variant_idx() -> None:
    k1 = cache_key(rule_id="r", input_text="a", variant_idx=0)
    k2 = cache_key(rule_id="r", input_text="a", variant_idx=1)
    assert k1 != k2


def test_cache_key_sensitive_to_temperature() -> None:
    k1 = cache_key(rule_id="r", input_text="a", temperature=0.0)
    k2 = cache_key(rule_id="r", input_text="a", temperature=0.5)
    assert k1 != k2


def test_cache_key_returns_hex_string() -> None:
    k = cache_key(rule_id="r", input_text="a")
    assert isinstance(k, str)
    assert len(k) == 64  # sha256 hex
    int(k, 16)  # valid hex, no exception


# -- ParaphraseCache disk round-trip -----------------------------------------


def test_cache_put_then_get_roundtrip(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    cache.put(
        rule_id="synonym-robustness",
        input_text="find a kettle",
        output="locate a teapot",
    )
    got = cache.get(rule_id="synonym-robustness", input_text="find a kettle")
    assert got == "locate a teapot"


def test_cache_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    got = cache.get(rule_id="synonym-robustness", input_text="never seen")
    assert got is None


def test_cache_survives_reload_across_instances(tmp_path: Path) -> None:
    """Disk persistence — a fresh ParaphraseCache with the same dir sees prior writes."""
    cache_a = ParaphraseCache(cache_dir=tmp_path)
    cache_a.put(rule_id="r", input_text="t", output="paraphrased t")
    cache_b = ParaphraseCache(cache_dir=tmp_path)
    assert cache_b.get(rule_id="r", input_text="t") == "paraphrased t"


def test_cache_separates_rules_by_file(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    cache.put(rule_id="synonym-robustness", input_text="x", output="y1")
    cache.put(rule_id="schema-paraphrase-invariance", input_text="x", output="y2")
    # Different rules have different JSONL files.
    files = {p.name for p in tmp_path.iterdir()}
    assert "synonym_robustness.jsonl" in files
    assert "schema_paraphrase_invariance.jsonl" in files


def test_cache_separates_variant_idx(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    cache.put(rule_id="r", input_text="t", output="v0", variant_idx=0)
    cache.put(rule_id="r", input_text="t", output="v1", variant_idx=1)
    assert cache.get(rule_id="r", input_text="t", variant_idx=0) == "v0"
    assert cache.get(rule_id="r", input_text="t", variant_idx=1) == "v1"


def test_cache_entry_roundtrip_via_dict() -> None:
    entry = CacheEntry(
        key="k", rule_id="r", input_text="t", output="o",
        variant_idx=0, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE,
        timestamp=1234.5,
    )
    again = CacheEntry.from_dict(entry.to_dict())
    assert again == entry


# -- paraphrase() orchestration ---------------------------------------------


def test_paraphrase_offline_miss_raises(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    with pytest.raises(ParaphraseCacheMiss):
        paraphrase(
            "find a kettle",
            rule_id="synonym-robustness",
            instruction="rephrase",
            cache=cache,
            offline=True,
        )


def test_paraphrase_offline_hit_returns_cached(tmp_path: Path) -> None:
    cache = ParaphraseCache(cache_dir=tmp_path)
    cache.put(rule_id="synonym-robustness", input_text="find a kettle", output="cached")
    got = paraphrase(
        "find a kettle",
        rule_id="synonym-robustness",
        instruction="rephrase",
        cache=cache,
        offline=True,
    )
    assert got == "cached"


def test_paraphrase_online_calls_gemini_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monkey-patched Gemini call — verifies the caching side effect."""
    cache = ParaphraseCache(cache_dir=tmp_path)
    calls = []

    def _fake_call_gemini(*, text, instruction, model, temperature):
        calls.append({"text": text, "instruction": instruction})
        return "gemini says: " + text

    monkeypatch.setattr(mod, "_call_gemini", _fake_call_gemini)

    # First call — hits Gemini, populates cache.
    out1 = paraphrase(
        "find a kettle",
        rule_id="synonym-robustness",
        instruction="rephrase",
        cache=cache,
    )
    assert out1 == "gemini says: find a kettle"
    assert len(calls) == 1

    # Second call with identical args — uses cache, no further Gemini call.
    out2 = paraphrase(
        "find a kettle",
        rule_id="synonym-robustness",
        instruction="rephrase",
        cache=cache,
    )
    assert out2 == out1
    assert len(calls) == 1  # still one


def test_paraphrase_module_has_no_top_level_google_generativeai_import() -> None:
    """The package must stay importable without `google-generativeai`.

    Only `_call_gemini()` may import it, and only on demand. This test
    inspects the module's source to confirm `google` / `google.generativeai`
    appears only inside the `_call_gemini` function body.

    Intentionally does NOT use `importlib.reload()` — reloading would
    rebind `ParaphraseCacheMiss` to a fresh class object, breaking
    `pytest.raises(ParaphraseCacheMiss)` in other test modules that
    captured the pre-reload class at their import time.
    """
    import inspect
    source = inspect.getsource(mod)
    # Find module-level imports (top of file, before any `def`/`class`).
    lines_before_first_def = []
    for line in source.splitlines():
        if line.lstrip().startswith(("def ", "class ")):
            break
        lines_before_first_def.append(line)
    module_header = "\n".join(lines_before_first_def)
    assert "google.generativeai" not in module_header
    assert "import google" not in module_header
    # Sanity: the dep IS referenced inside _call_gemini (lazy import).
    assert "google.generativeai" in source  # somewhere — inside the function
