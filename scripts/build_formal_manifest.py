"""Build the immutable plan for the formal SAGE reproduction.

The paper does not publish its 120 feature IDs or sampling seed. This script
therefore keeps the paper-published qualitative features as canaries and builds
a separate deterministic 120-feature matrix for the aggregate reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260729
OFFICIAL_PAPER_CODE_COMMIT = "add7ed7331e3d0d6fb497b2eb806aea4bc07c503"

TARGETS = [
    {
        "target_llm": "Qwen/Qwen3-4B",
        "neuronpedia_model_id": "qwen3-4b",
        "base_model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "source_template": "{layer}-transcoder-hp",
        "d_sae": 163840,
        "activation_backend": "local_saelens",
        "saelens_release": "mwhanna-qwen3-4b-transcoders",
        "saelens_sae_id_template": "layer_{layer}",
        "huggingface_repo": "mwhanna/qwen3-4b-transcoders",
        "neuronpedia_inference_enabled": False,
        "sae_revision": "94d176260ac39ce2f882b8b09aba8c118df29bb3",
    },
    {
        "target_llm": "google/gemma-2-2b",
        "neuronpedia_model_id": "gemma-2-2b",
        "base_model_revision": "c5ebcd40d208330abc697524c919956e692655cf",
        "source_template": "{layer}-gemmascope-res-16k",
        "d_sae": 16384,
        "activation_backend": "neuronpedia_api",
        "saelens_release": "gemma-scope-2b-pt-res-canonical",
        "saelens_sae_id_template": "layer_{layer}/width_16k/canonical",
        "neuronpedia_inference_enabled": True,
    },
    {
        "target_llm": "openai/gpt-oss-20b",
        "neuronpedia_model_id": "gpt-oss-20b",
        "base_model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
        "source_template": "{layer}-resid-post-aa",
        "d_sae": 131072,
        "activation_backend": "neuronpedia_api",
        "neuronpedia_inference_enabled": True,
    },
]

PUBLISHED_CANARIES = [
    {"model": "gemma-2-2b", "layer": 11, "feature": 5125, "source": "11-gemmascope-res-16k"},
    {"model": "gemma-2-2b", "layer": 11, "feature": 13574, "source": "11-gemmascope-res-16k"},
    {"model": "qwen3-4b", "layer": 3, "feature": 148551, "source": "3-transcoder-hp"},
    {"model": "qwen3-4b", "layer": 7, "feature": 158076, "source": "7-transcoder-hp"},
    {"model": "qwen3-4b", "layer": 23, "feature": 24625, "source": "23-transcoder-hp"},
    {"model": "gpt-oss-20b", "layer": 3, "feature": 72038, "source": "3-resid-post-aa"},
    {"model": "gpt-oss-20b", "layer": 3, "feature": 121075, "source": "3-resid-post-aa"},
]


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def working_diff_sha256() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    hasher = hashlib.sha256()
    hasher.update(result.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    source_roots = {
        "core",
        "environment",
        "scripts",
        "tests",
        "tools",
        "utils",
    }
    for raw_path in filter(None, untracked.stdout.split(b"\0")):
        relative = Path(raw_path.decode("utf-8"))
        if relative.name not in {"main.py", ".gitignore"} and relative.parts[0] not in source_roots:
            continue
        hasher.update(b"\0untracked\0" + raw_path + b"\0")
        hasher.update((ROOT / relative).read_bytes())
    return hasher.hexdigest()


def source_seed(base_seed: int, model_id: str, layer: int) -> int:
    payload = f"{base_seed}|{model_id}|{layer}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_matrix(base_seed: int) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for target in TARGETS:
        for layer in (3, 7, 11, 23):
            seed = source_seed(base_seed, target["neuronpedia_model_id"], layer)
            features = random.Random(seed).sample(range(target["d_sae"]), 10)
            matrix.append(
                {
                    "target_llm": target["target_llm"],
                    "neuronpedia_model_id": target["neuronpedia_model_id"],
                    "layer": layer,
                    "source": target["source_template"].format(layer=layer),
                    "d_sae": target["d_sae"],
                    "activation_backend": target["activation_backend"],
                    "source_seed": seed,
                    "features": features,
                }
            )
    return matrix


def build_manifest(base_seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "formal_aggregate_running",
        "paper": {
            "title": "SAGE: An Agentic Explainer Framework for Interpreting SAE Features in Language Models",
            "venue": "EACL 2026 Industry Track",
            "doi": "10.18653/v1/2026.eacl-industry.37",
            "acl_url": "https://aclanthology.org/2026.eacl-industry.37/",
        },
        "code": {
            "official_repository": "https://github.com/jiujiubuhejiu/SAGE",
            "official_paper_code_commit": OFFICIAL_PAPER_CODE_COMMIT,
            "local_head": git_output("rev-parse", "HEAD"),
            "upstream_main": git_output("rev-parse", "upstream/main"),
            "working_diff_sha256": working_diff_sha256(),
        },
        "paper_main_spec": {
            "agent_model_family": "gpt-5",
            "layers": [3, 7, 11, 23],
            "features_per_layer": 10,
            "target_models": 3,
            "total_features": 120,
            "top_k_exemplars": 10,
            "initial_hypotheses": 4,
            "max_rounds": 14,
            "generated_examples_per_explanation": 10,
            "generation_success_threshold": "0.5 * max(top_10_exemplar_activations)",
            "prediction_metric": "mean per-example Pearson correlation",
            "prediction_scale": "integer activation 0-10 from token logprobs",
        },
        "replication_pins": {
            "agent_and_generation_model": "gpt-5-2025-08-07",
            "prediction_simulator_model": "gpt-4o-2024-11-20",
            "base_random_seed": base_seed,
            "note": "Snapshots and seed are replication choices because the paper does not publish them.",
        },
        "data_policy": {
            "source": "live public Neuronpedia feature and activation APIs",
            "freeze": "raw feature/baseline responses, exemplars, selected indices, retrieval timestamp, and SHA256 are saved per feature",
            "baseline": "prefer exact existing gpt-5/oai_token-act-pair; otherwise generate locally with the official Neuronpedia prompt port and pinned GPT-5 snapshot",
            "remote_mutation": "disabled; no explanation delete, generate, or upload calls",
        },
        "targets": TARGETS,
        "paper_published_canaries": PUBLISHED_CANARIES,
        "aggregate_matrix": build_matrix(base_seed),
        "known_non_recoverable_paper_details": [
            "The paper does not publish the complete 120 feature IDs.",
            "The paper does not publish the feature-sampling seed.",
            "The paper does not pin GPT-5 or prediction-simulator snapshots.",
            "The paper does not freeze its Neuronpedia data snapshot or held-out exemplar indices.",
            "The official repository has no paper-version tag or released aggregate result artifacts.",
        ],
        "execution_gate": {
            "canaries_before_aggregate": True,
            "qwen_requires_local_model_and_transcoder": True,
            "stop_on_metric_or_data_contract_failure": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reproduction" / "formal_manifest.json"
    )
    args = parser.parse_args()
    manifest = build_manifest(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"aggregate_sources={len(manifest['aggregate_matrix'])}")
    print(f"aggregate_features={sum(len(x['features']) for x in manifest['aggregate_matrix'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
