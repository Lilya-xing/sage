"""Live Neuronpedia API smoke test for one paper qualitative feature."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from core.system import System
from tools.neuronpedia import NeuronpediaManager


SMOKE_BANNER = "SMOKE TEST ONLY — NOT A PAPER RESULT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="gemma-2-2b")
    parser.add_argument("--source", default="11-gemmascope-res-16k")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--feature", type=int, default=5125)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--text",
        default="Use threading.Lock() to protect the shared queue.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    print(f"=== {SMOKE_BANNER} ===")
    system = System(
        llm_name=args.model_id,
        sae_path="api-mode",
        sae_layer=args.layer,
        feature_index=args.feature,
        device="cpu",
        use_api_for_activations=True,
        neuronpedia_model_id=args.model_id,
        neuronpedia_source=args.source,
    )
    manager = NeuronpediaManager(
        system=system,
        use_api_for_activations=True,
        neuronpedia_model_id=args.model_id,
        neuronpedia_source=args.source,
    )

    exemplars = manager.get_maximally_activating_examples_from_api(
        top_k=args.top_k,
        return_detailed=True,
    )
    if len(exemplars) != args.top_k:
        raise RuntimeError(
            f"expected {args.top_k} exemplars, received {len(exemplars)}"
        )
    if any(len(item["tokens"]) != len(item["per_token_activations"]) for item in exemplars):
        raise RuntimeError("exemplar token/value lengths do not match")

    trace = system.get_activation_trace(args.text)
    tokens = trace["tokens"]
    values = trace["per_token_activation"]
    maximum = float(trace["summary_activation"])
    if not tokens or len(tokens) != len(values):
        raise RuntimeError("custom-text trace is empty or misaligned")
    if not math.isfinite(maximum):
        raise RuntimeError("custom-text activation is not finite")

    summary = {
        "test_kind": "smoke",
        "paper_result": False,
        "model_id": args.model_id,
        "source": args.source,
        "layer": args.layer,
        "feature": args.feature,
        "top_k_received": len(exemplars),
        "top_exemplar_max_activation": max(
            float(item["max_activation"]) for item in exemplars
        ),
        "custom_text": args.text,
        "custom_text_max_activation": maximum,
        "custom_text_max_token_index": trace["max_token_index"],
        "custom_text_token_count": len(tokens),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

