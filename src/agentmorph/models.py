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


def _fold_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Fold every `system` message into the content of the next `user` message.

    Gemma-family chat templates reject the `system` role. Instead of
    special-casing Gemma everywhere upstream, we reshape the message list at
    the tokenizer boundary. Semantics are preserved: the system instructions
    become the prefix of the user turn they were meant to condition.
    """
    out: list[dict[str, str]] = []
    pending: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "system":
            pending.append(content)
            continue
        if role == "user" and pending:
            merged = "\n\n".join(pending + [content])
            out.append({"role": "user", "content": merged})
            pending = []
        else:
            out.append({"role": role, "content": content})
    if pending:
        # No user turn followed — prepend the accumulated system text as a
        # standalone user message so the generation still sees it.
        out.insert(0, {"role": "user", "content": "\n\n".join(pending)})
    return out


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

        # Gemma-2's chat template rejects the "system" role outright
        # (`TemplateError: System role not supported`). Llama, Qwen, and Phi
        # accept it. Try once with the message list as-is; on a chat-template
        # error, fold system messages into the first user message and retry.
        try:
            template_out = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception as exc:
            err = str(exc)
            if "System role" in err or "system role" in err or "Only user and model roles" in err:
                folded = _fold_system_into_user(messages)
                template_out = tok.apply_chat_template(
                    folded,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                raise

        # transformers 4.45+ changed `apply_chat_template(return_tensors="pt")`
        # to return a BatchEncoding (dict-like) rather than a bare Tensor for
        # most tokenizers. Older transformers returned a Tensor. Handle both.
        # The original untraceable "AttributeError:" in Stage-3 came from
        # passing a BatchEncoding positionally to `.generate()`, which tried
        # to access a Tensor-only attribute on the dict wrapper.
        if isinstance(template_out, torch.Tensor):
            input_ids = template_out
            attention_mask = torch.ones_like(input_ids)
        else:
            # BatchEncoding or plain dict from newer transformers.
            input_ids = template_out["input_ids"]
            if "attention_mask" in template_out:
                attention_mask = template_out["attention_mask"]
            else:
                attention_mask = torch.ones_like(input_ids)

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # Sampling config:
        #   * Greedy (temperature == 0) → do_sample=False, omit temperature
        #     entirely. transformers >=4.50 rejects temperature / top_p
        #     when do_sample=False ("generation flags are not valid and
        #     may be ignored").
        #   * Sampling (temperature > 0) → do_sample=True, pass temperature
        #     through.
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out_ids = self.model.generate(
                input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        new_tokens = out_ids[0, input_ids.shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)

        if stop:
            # Soft stop — trim at first occurrence of any stop string.
            cut = min((text.find(s) for s in stop if text.find(s) >= 0), default=-1)
            if cut >= 0:
                text = text[:cut]
        return text.strip()


_LOAD_CACHE: dict[str, LoadedModel] = {}


def _choose_load_profile(quantize: bool) -> tuple[str, bool, str]:
    """Decide (device, quantize_4bit, dtype_name) based on what's available.

    GPU is the preferred path. If CUDA isn't available, we fall back to CPU
    with no quantization (bitsandbytes requires CUDA) and fp32 compute.
    CPU inference is ~100x slower than T4 — only useful for development,
    unit tests, or tiny-model smoke runs. The runner will still work; it
    just takes minutes per token instead of seconds.
    """
    import torch
    if torch.cuda.is_available():
        # Preferred path: CUDA + 4-bit nf4 quantization.
        return "cuda", bool(quantize), "float16"
    # Fallback: CPU, no bnb quantization, fp32 (fp16 on CPU is unstable).
    return "cpu", False, "float32"


def load_model(
    model_id: str,
    *,
    hf_cache_dir: str | None = None,
    reuse_cached: bool = True,
    quantize: bool = True,
    force_cpu: bool = False,
) -> LoadedModel:
    """Load a primary model.

    GPU is the preferred path — 4-bit nf4 quantization on CUDA. The function
    automatically falls back to CPU (fp32, no quantization) if CUDA isn't
    available, or if `force_cpu=True`. CPU inference is far slower and uses
    ~4x more memory than 4-bit GPU, but the entire pipeline still runs,
    which lets you exercise adapters, mutation rules, and the runner end-
    to-end without a GPU for development.

    `reuse_cached=True` returns the same `LoadedModel` for repeated calls in
    one session — important because Colab T4s can't hold two 7B+ models in
    memory at once.
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

    # Choose load profile — GPU preferred, CPU fallback.
    if force_cpu:
        device, do_quant, dtype_name = "cpu", False, "float32"
    else:
        device, do_quant, dtype_name = _choose_load_profile(quantize)

    if device == "cpu":
        print(
            f"[agentmorph.models] No CUDA detected; loading {spec.id} on CPU "
            f"in {dtype_name} without bitsandbytes quantization. "
            f"Inference will be ~100x slower than a T4 — use for development "
            f"and small-model smoke runs only."
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        if do_quant:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.float16

        # VRAM budget: cap GPU usage so transformers spills layers to CPU
        # instead of OOM-ing during the materialisation phase.
        #
        # First-pass headroom of 1.5 GiB was insufficient: transformers'
        # `core_model_loading` materialises tensors in parallel, and each
        # in-flight tensor briefly holds an fp16 staging buffer BEFORE the
        # 4-bit quantisation runs. The peak crossed our budget and Phi-4
        # OOMed at 56 % loaded on a 14.56 GiB T4. Bumping headroom to
        # 4.5 GiB forces transformers to place more weights on CPU
        # up-front, leaving enough room for the staging peak.
        #
        # Smaller models (Llama-3.2-3B at ~2.5 GB, Qwen-7B at ~4 GB,
        # Llama-3.1-8B at ~4.5 GB, Gemma-2-9B at ~5.5 GB) all fit well
        # under a 10 GiB GPU budget — so on T4 they stay fully on GPU
        # with zero perf cost. Only Phi-4 (~7-8 GB at 4-bit, plus the
        # higher load peak) gets partial CPU offload.
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        budget_gb = max(total_gb - 4.5, 4.0)
        model_kwargs["max_memory"] = {
            0: f"{budget_gb:.1f}GiB",
            "cpu": "30GiB",
        }

        # Disk offload backup: if even the CPU-offloaded layers spill the
        # 25 GB Colab CPU RAM, transformers will use this folder instead of
        # OOMing CPU-side. `offload_state_dict=True` also forces the loader
        # to stream the state dict from disk one piece at a time, lowering
        # the materialisation peak. Necessary for Phi-4 on T4; harmless on
        # smaller models (the folder stays empty).
        import tempfile
        offload_dir = os.path.join(tempfile.gettempdir(), "agentmorph_offload")
        os.makedirs(offload_dir, exist_ok=True)
        model_kwargs["offload_folder"] = offload_dir
        model_kwargs["offload_state_dict"] = True
    else:
        # CPU: no device_map (accelerate behavior varies on CPU), explicit
        # fp32 compute, no bnb. `.to("cpu")` is implicit after load.
        model_kwargs["torch_dtype"] = torch.float32

    # Empty the CUDA cache right before from_pretrained — the materialisation
    # path is sensitive to fragmentation and a fresh allocator state buys us
    # the few hundred MB that pushed Phi-4 over the edge on T4.
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(spec.hf_repo, **model_kwargs)
    model.eval()

    # Ensure we actually know where the model landed.
    actual_device = "cuda" if next(model.parameters()).is_cuda else "cpu"

    loaded = LoadedModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=actual_device,
        dtype=dtype_name,
        quantization="nf4-4bit" if (device == "cuda" and do_quant) else dtype_name,
    )
    if reuse_cached:
        _LOAD_CACHE[model_id] = loaded
    return loaded


def unload_model(model_id: str) -> None:
    """Drop the cached model and free memory.

    Call this between primary models in the baseline runner — T4 VRAM is too
    small to hold two 7B+ models simultaneously. On CPU this still runs gc
    to release RAM between models.
    """
    loaded = _LOAD_CACHE.pop(model_id, None)
    if loaded is None:
        return
    try:
        import gc
        del loaded.model
        del loaded.tokenizer
        gc.collect()
        # Only touch CUDA if it's actually available — CPU-only runs would
        # otherwise trip `torch.cuda.empty_cache()` with no device.
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_cache() -> None:
    for mid in list(_LOAD_CACHE):
        unload_model(mid)
