"""Run the frozen 40-feature Qwen matrix with API exemplars and local traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qwen_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in manifest["aggregate_matrix"]
        if entry["neuronpedia_model_id"] == "qwen3-4b"
    ]
    entries.sort(key=lambda entry: entry["layer"])
    if [entry["layer"] for entry in entries] != [3, 7, 11, 23]:
        raise ValueError("manifest must contain Qwen layers 3, 7, 11, and 23")
    if sum(len(entry["features"]) for entry in entries) != 40:
        raise ValueError("manifest must contain exactly 40 Qwen features")
    return entries


def build_command(
    manifest: dict[str, Any], output_root: Path, device: str
) -> list[str]:
    entries = qwen_entries(manifest)
    target = next(
        target
        for target in manifest["targets"]
        if target["neuronpedia_model_id"] == "qwen3-4b"
    )
    spec = manifest["paper_main_spec"]
    pins = manifest["replication_pins"]
    features = ";".join(
        f"layer{entry['layer']}={','.join(map(str, entry['features']))}"
        for entry in entries
    )
    return [
        sys.executable,
        str(ROOT / "main.py"),
        "--agent_llm",
        pins["agent_and_generation_model"],
        "--target_llm",
        target["target_llm"],
        "--model_revision",
        target["base_model_revision"],
        "--sae_path",
        (
            "sae-lens://release="
            f"{target['saelens_release']};sae_id=layer_{{layer}}"
        ),
        "--features",
        features,
        "--path2save",
        str(output_root),
        "--max_rounds",
        str(spec["max_rounds"]),
        "--timeout_minutes",
        "30",
        "--top_k",
        str(spec["top_k_exemplars"]),
        "--initial_hypotheses",
        str(spec["initial_hypotheses"]),
        "--use_api_for_activations",
        "false",
        "--use_api_for_exemplars",
        "true",
        "--use_saedashboard",
        "false",
        "--neuronpedia_model_id",
        "qwen3-4b",
        "--neuronpedia_source",
        "{layer}-transcoder-hp",
        "--device",
        device,
    ]


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reproduction" / "formal_manifest.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "formal-eacl2026"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    command = build_command(manifest, args.output_root, args.device)
    state_path = ROOT / "reproduction" / "formal_runner_qwen3_4b.json"
    state = {
        "target": "qwen3-4b",
        "manifest": str(args.manifest),
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "command": command,
    }
    write_state(state_path, state)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=ROOT, check=False)
    state["returncode"] = completed.returncode
    state["status"] = "completed" if completed.returncode == 0 else "failed"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
