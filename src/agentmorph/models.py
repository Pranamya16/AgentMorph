"""4-bit quantized model loader + registry for the 5 Stage 1 primary models.

All 5 models run on a Colab T4 via bitsandbytes nf4 quantization. This module
is the single place that knows about `transformers`, `torch`, and
`BitsAndBytesConfig` — everything else in the package treats a loaded model
as an opaque `LoadedModel` handle with a `generate()` method.

Design notes:
  * Lazy imports of `torch` / `transformers` so `pip install agentmorph`
    without the `[models]` extra still works (for CI, docs, unit tests).
  * `MODEL_REGISTRY` is the authoritative list of primary models from the
    execution plan. Do not add/remove entries without updating CLAUDE.md.
  * `hf_cache_dir` defaults to `/content/hf_cache` on Colab (mount it on
    Google Drive so model weights survive a session kill — see the Colab
    bootstrap script).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Opt into PyTorch's expandable-segments allocator before torch is imported
# anywhere else. This dramatically reduces VRAM fragmentation on long runs
# where we free/reallocate between scenarios — exactly our Stage 1 shape.
# The CUDA OOM error that hit the first smolagents sweep recommended this
# explicitly.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if TYPE_CHECKING:  # pragma: no cover — typing only
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# -- Registry ---------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A model we can load locally on Colab T4 in 4-bit."""

    id: str                           # short name we use everywhere
    hf_repo: str                      # HuggingFace repo id
    chat_template: str = "auto"       # "auto" = use tokenizer's built-in
    trust_remote_code: bool = False
    # Approximate GPU memory at 4-bit in GB — sanity check before loading.
    approx_vram_gb: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        ModelSpec(
            id="Llama-3.2-3B",
            hf_repo="meta-llama/Llama-3.2-3B-Instruct",
            approx_vram_gb=2.5,
            tags=("llama", "small"),
        ),
        ModelSpec(
            id="Qwen2.5-7B",
            hf_repo="Qwen/Qwen2.5-7B-Instruct",
            approx_vram_gb=5.5,
            tags=("qwen",),
        ),
        ModelSpec(
            id="Gemma-2-9B",
            hf_repo="google/gemma-2-9b-it",
            approx_vram_gb=7.0,
            tags=("gemma",),
        ),
        ModelSpec(
            id="Phi-4",
            hf_repo="microsoft/phi-4",
            trust_remote_code=True,
            approx_vram_gb=10.5,
            tags=("phi", "tight-on-t4"),
        ),
        ModelSpec(
            id="Llama-3.1-8B",
            hf_repo="meta-llama/Llama-3.1-8B-Instruct",
            approx_vram_gb=6.0,
            tags=("llama",),
        ),
    ]
}

PRIMARY_MODEL_IDS: tuple[str, ...] = tuple(MODEL_REGISTRY.keys())


def get_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_id!r}. Known: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_id]


# -- Loader -----------------------------------------------------------------


@dataclass
class LoadedModel:
    """Framework-agnostic handle returned by `load_model`.

    Both the smolagents and LangGraph adapters read the same fields off this.
    """

    spec: ModelSpec
    model: Any                        # transformers.PreTrainedModel
    tokenizer: Any                    # transformers.PreTrainedTokenizerBase
    device: str = "cuda"
    dtype: str = "float16"
    quantization: str = "nf4-4bit"

    @property
    def id(self) -> str:
        return self.spec.id

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Apply chat template, run a single generate, return the new text."""
        import torch  # lazy

        tok = self.tokenizer
        prompt_ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )

        with torch.inference_mode():
            out_ids = self.model.generate(prompt_ids, **gen_kwargs)

        new_tokens = out_ids[0, prompt_ids.shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)

        if stop:
            # Soft stop — trim at first occurrence of any stop string.
            cut = min((text.find(s) for s in stop if text.find(s) >= 0), default=-1)
            if cut >= 0:
                text = text[:cut]
        return text.strip()


_LOAD_CACHE: dict[str, LoadedModel] = {}


def load_model(
    model_id: str,
    *,
    hf_cache_dir: str | None = None,
    reuse_cached: bool = True,
    quantize: bool = True,
) -> LoadedModel:
    """Load a primary model in 4-bit on the current CUDA device.

    `reuse_cached=True` returns the same `LoadedModel` for repeated calls in
    one session — important because Colab T4s can't hold two 7B+ models in
    memory at once. The baseline runner relies on this.

    If `quantize=False`, falls back to fp16 (only viable for Llama-3.2-3B).
    """
    if reuse_cached and model_id in _LOAD_CACHE:
        return _LOAD_CACHE[model_id]

    spec = get_spec(model_id)

    # Lazy imports — keep package importable without the [models] extra.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = hf_cache_dir or os.environ.get("AGENTMORPH_HF_CACHE") or None

    tok_kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "device_map": "auto",
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    if quantize:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(spec.hf_repo, **model_kwargs)
    model.eval()

    loaded = LoadedModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=str(getattr(model, "device", "cuda")),
        dtype="float16",
        quantization="nf4-4bit" if quantize else "fp16",
    )
    if reuse_cached:
        _LOAD_CACHE[model_id] = loaded
    return loaded


def unload_model(model_id: str) -> None:
    """Drop the cached model and free GPU memory.

    Call this between primary models in the baseline runner — T4 VRAM is too
    small to hold two 7B+ models simultaneously.
    """
    loaded = _LOAD_CACHE.pop(model_id, None)
    if loaded is None:
        return
    try:
        import torch
        del loaded.model
        del loaded.tokenizer
        torch.cuda.empty_cache()
    except Exception:
        pass


def clear_cache() -> None:
    for mid in list(_LOAD_CACHE):
        unload_model(mid)
