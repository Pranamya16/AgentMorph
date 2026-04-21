"""Tests for the CPU fallback in `agentmorph.models`.

These don't load a real model — they exercise the branching logic of
`_choose_load_profile` and verify `unload_model` / `_reclaim_vram` are safe
when no CUDA is present.

Skipped entirely if `torch` isn't installed (matches the convention the rest
of the suite follows — the package is importable without the `[models]`
extra, and so should the tests be runnable without it).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip the module if torch missing


@pytest.fixture
def mock_no_cuda(monkeypatch: pytest.MonkeyPatch):
    """Pretend `torch.cuda.is_available()` returns False."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield


@pytest.fixture
def mock_cuda(monkeypatch: pytest.MonkeyPatch):
    """Pretend `torch.cuda.is_available()` returns True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    yield


def test_choose_load_profile_prefers_gpu_when_cuda_available(mock_cuda) -> None:
    from agentmorph.models import _choose_load_profile
    device, quantize, dtype = _choose_load_profile(quantize=True)
    assert device == "cuda"
    assert quantize is True
    assert dtype == "float16"


def test_choose_load_profile_respects_quantize_false_on_gpu(mock_cuda) -> None:
    from agentmorph.models import _choose_load_profile
    device, quantize, dtype = _choose_load_profile(quantize=False)
    assert device == "cuda"
    assert quantize is False
    assert dtype == "float16"


def test_choose_load_profile_falls_back_to_cpu_fp32_when_no_cuda(mock_no_cuda) -> None:
    from agentmorph.models import _choose_load_profile
    device, quantize, dtype = _choose_load_profile(quantize=True)
    assert device == "cpu"
    # bnb quantization requires CUDA — must disable on CPU regardless of flag.
    assert quantize is False
    # fp16 on CPU is unstable on many torch versions; fp32 is the safe choice.
    assert dtype == "float32"


def test_choose_load_profile_cpu_ignores_quantize_flag(mock_no_cuda) -> None:
    """Even if the caller requests quantization, CPU must refuse — bnb will crash."""
    from agentmorph.models import _choose_load_profile
    _, quantize, _ = _choose_load_profile(quantize=True)
    assert quantize is False


def test_unload_model_safe_without_cuda(mock_no_cuda) -> None:
    """unload_model must not call torch.cuda.empty_cache when no CUDA is present."""
    from agentmorph import models
    # Simulate a cached entry without actually loading a model.
    class _Sentinel:
        model = object()
        tokenizer = object()
    models._LOAD_CACHE["test-entry"] = _Sentinel()  # type: ignore[assignment]
    # Should not raise even though CUDA is "unavailable".
    models.unload_model("test-entry")
    assert "test-entry" not in models._LOAD_CACHE


def test_unload_model_noop_on_missing_key() -> None:
    from agentmorph.models import unload_model
    # No entry — must not raise.
    unload_model("nonexistent-model-id")


def test_reclaim_vram_safe_without_cuda(mock_no_cuda) -> None:
    from agentmorph.runner import _reclaim_vram
    # Should silently succeed without CUDA.
    _reclaim_vram()


def test_reclaim_vram_safe_with_cuda(mock_cuda) -> None:
    """With a mocked is_available=True, torch.cuda.empty_cache might raise
    if there's no real device. The helper swallows those errors."""
    from agentmorph.runner import _reclaim_vram
    _reclaim_vram()  # must not raise


def test_load_model_registry_unchanged() -> None:
    """Stage 1 invariant: the 5 primary models stay in the registry."""
    from agentmorph.models import MODEL_REGISTRY, PRIMARY_MODEL_IDS
    assert set(PRIMARY_MODEL_IDS) == {
        "Llama-3.2-3B", "Qwen2.5-7B", "Gemma-2-9B", "Phi-4", "Llama-3.1-8B"
    }
    assert len(MODEL_REGISTRY) == 5
