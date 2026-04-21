"""One-shot Gemini paraphrase-cache seeding for Stage 2 rules 2, 3, 5.

Run once before the Stage 3 sweep:

    export GEMINI_API_KEY=...
    python scripts/generate_paraphrases.py
    git add runs/paraphrase_cache/
    git commit -m "paraphrase: seed cache for Stage 2 rules 2, 3, 5"
    git push

The sweep then reads the cache in `offline=True` mode and never hits the
Gemini API during a potentially-long run.

Total API calls seeded:
  * Rule 2 (schema-paraphrase-invariance):  30 (one per tool description)
  * Rule 3 (synonym-robustness):            20 (one per seed scenario)
  * Rule 5 (refusal-consistency):            4 (2 scenarios × 2 variants)
  ---------------------------------------------------------------
  Total:                                    54  (well under free-tier 1500/day)

If Gemini returns a degenerate output (empty or identical to input), the
script retries once at temperature=0.5. Past that, it logs a warning and
writes a template fallback so the cache is never missing a key.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmorph.environments.ecommerce_env import EcommerceEnvironment
from agentmorph.paraphrase import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ParaphraseCache,
    paraphrase,
)
from agentmorph.rules.refusal_consistency import (
    NUM_VARIANTS as REFUSAL_VARIANTS,
    PARAPHRASE_INSTRUCTION as REFUSAL_INSTRUCTION,
)
from agentmorph.rules.schema_paraphrase_invariance import (
    PARAPHRASE_INSTRUCTION as SCHEMA_INSTRUCTION,
)
from agentmorph.rules.synonym_robustness import (
    PARAPHRASE_INSTRUCTION as SYNONYM_INSTRUCTION,
)
from agentmorph.tools.ecommerce import build_ecommerce_registry


def _seed_one(
    *,
    cache: ParaphraseCache,
    rule_id: str,
    instruction: str,
    input_text: str,
    variant_idx: int = 0,
    dry_run: bool = False,
) -> None:
    """Populate one cache entry, retrying once on degenerate output."""
    # Already cached? Skip silently (idempotent seeding).
    if cache.get(
        rule_id=rule_id,
        input_text=input_text,
        variant_idx=variant_idx,
    ) is not None:
        print(f"  [skip] {rule_id} variant={variant_idx} already cached")
        return

    if dry_run:
        print(f"  [DRY] would paraphrase: {input_text[:60]!r}")
        return

    # First attempt at default temperature.
    try:
        output = paraphrase(
            input_text,
            rule_id=rule_id,
            instruction=instruction,
            variant_idx=variant_idx,
            cache=cache,
            offline=False,
            temperature=DEFAULT_TEMPERATURE,
        )
    except Exception as exc:
        print(f"  [WARN] Gemini call failed ({type(exc).__name__}: {exc}); writing template fallback")
        _write_template_fallback(
            cache, rule_id=rule_id, input_text=input_text, variant_idx=variant_idx
        )
        return

    # Degenerate-output check (empty or identical to input).
    if output and output.strip() != input_text.strip() and len(output) > 5:
        print(f"  [ok]   variant={variant_idx} -> {output[:60]!r}")
        return

    # Retry once at a higher temperature for more variety.
    print("  [retry] degenerate output; retrying at temperature=0.5")
    try:
        output = paraphrase(
            input_text,
            rule_id=rule_id,
            instruction=instruction,
            variant_idx=variant_idx,
            cache=cache,
            offline=False,
            temperature=0.5,
        )
    except Exception as exc:
        print(f"  [WARN] retry failed: {exc}; writing template fallback")
        _write_template_fallback(
            cache, rule_id=rule_id, input_text=input_text, variant_idx=variant_idx
        )
        return

    if output and output.strip() != input_text.strip() and len(output) > 5:
        print(f"  [ok*]  variant={variant_idx} -> {output[:60]!r} (after retry)")
    else:
        print("  [WARN] still degenerate after retry; writing template fallback")
        _write_template_fallback(
            cache, rule_id=rule_id, input_text=input_text, variant_idx=variant_idx
        )


def _write_template_fallback(
    cache: ParaphraseCache,
    *,
    rule_id: str,
    input_text: str,
    variant_idx: int,
) -> None:
    """Minimal deterministic rephrase so the cache is never short."""
    if rule_id == "schema-paraphrase-invariance":
        template = f"Alternative phrasing: {input_text}"
    elif rule_id == "synonym-robustness":
        template = f"Could you help me with the following? {input_text}"
    elif rule_id == "refusal-consistency":
        template = f"{input_text} (variant {variant_idx})"
    else:
        template = input_text
    cache.put(
        rule_id=rule_id,
        input_text=input_text,
        output=template,
        variant_idx=variant_idx,
    )


def seed_rule_2(cache: ParaphraseCache, *, dry_run: bool = False) -> int:
    """Schema-paraphrase: 1 variant × 30 tool descriptions = 30 entries."""
    print("\n=== Rule 2 — schema-paraphrase-invariance ===")
    _state, registry = build_ecommerce_registry()
    count = 0
    for tool in registry:
        _seed_one(
            cache=cache,
            rule_id="schema-paraphrase-invariance",
            instruction=SCHEMA_INSTRUCTION,
            input_text=tool.description,
            variant_idx=0,
            dry_run=dry_run,
        )
        count += 1
    return count


def seed_rule_3(cache: ParaphraseCache, *, dry_run: bool = False) -> int:
    """Synonym-robustness: 1 variant × 20 seed scenarios = 20 entries."""
    print("\n=== Rule 3 — synonym-robustness ===")
    env = EcommerceEnvironment()
    count = 0
    for scenario in env.scenarios():
        _seed_one(
            cache=cache,
            rule_id="synonym-robustness",
            instruction=SYNONYM_INSTRUCTION,
            input_text=scenario.prompt,
            variant_idx=0,
            dry_run=dry_run,
        )
        count += 1
    return count


def seed_rule_5(cache: ParaphraseCache, *, dry_run: bool = False) -> int:
    """Refusal-consistency: 2 variants × 2 refusal scenarios = 4 entries."""
    print("\n=== Rule 5 — refusal-consistency ===")
    env = EcommerceEnvironment()
    count = 0
    for scenario in env.scenarios():
        if not scenario.metadata.get("should_refuse"):
            continue
        for variant_idx in range(REFUSAL_VARIANTS):
            _seed_one(
                cache=cache,
                rule_id="refusal-consistency",
                instruction=REFUSAL_INSTRUCTION,
                input_text=scenario.prompt,
                variant_idx=variant_idx,
                dry_run=dry_run,
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/paraphrase_cache"),
        help="Where the paraphrase cache JSONL files live (default: runs/paraphrase_cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip live Gemini calls — just report what would be seeded.",
    )
    parser.add_argument(
        "--only",
        choices=["rule2", "rule3", "rule5"],
        default=None,
        help="Seed only one rule (default: seed all three).",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY environment variable is not set. Get a key at "
            "https://aistudio.google.com/app/apikey and `export GEMINI_API_KEY=...`. "
            "Pass --dry-run to preview without calling the API.",
            file=sys.stderr,
        )
        return 1

    cache = ParaphraseCache(cache_dir=args.cache_dir)
    total = 0

    if args.only in (None, "rule2"):
        total += seed_rule_2(cache, dry_run=args.dry_run)
    if args.only in (None, "rule3"):
        total += seed_rule_3(cache, dry_run=args.dry_run)
    if args.only in (None, "rule5"):
        total += seed_rule_5(cache, dry_run=args.dry_run)

    print(f"\nDone — seeded {total} paraphrase entries at {args.cache_dir}")
    print(f"Using Gemini model: {DEFAULT_MODEL}")
    if not args.dry_run:
        print("Next: `git add runs/paraphrase_cache/ && git commit -m "
              "\"paraphrase: seed cache for Stage 2 rules 2, 3, 5\" && git push`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
