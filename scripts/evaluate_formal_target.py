"""Evaluate one target's frozen formal matrix after SAGE generation finishes."""

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


def generation_result_valid(path: Path) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(
        "[DESCRIPTION]:" in analysis
        and "[EVIDENCE]:" in analysis
        and "[LABEL" in analysis
        for analysis in result.get("analysis_history", [])
    )


def target_entries(
    manifest: dict[str, Any], target_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = next(
        item for item in manifest["targets"]
        if item["neuronpedia_model_id"] == target_id
    )
    entries = [
        item for item in manifest["aggregate_matrix"]
        if item["neuronpedia_model_id"] == target_id
    ]
    entries.sort(key=lambda item: item["layer"])
    if [item["layer"] for item in entries] != [3, 7, 11, 23]:
        raise ValueError(f"incomplete four-layer matrix for {target_id}")
    return target, entries


def build_command(
    manifest: dict[str, Any],
    target: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    device: str,
) -> list[str]:
    pins = manifest["replication_pins"]
    layer_dir = (
        output_root
        / pins["agent_and_generation_model"]
        / target["target_llm"].replace("/", "_")
        / f"layer_{entry['layer']}"
    )
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--sage_results_path",
        str(layer_dir),
        "--features",
        ",".join(map(str, entry["features"])),
        "--model_name",
        target["target_llm"],
        "--layer",
        entry["source"],
        "--neuronpedia_model_id",
        target["neuronpedia_model_id"],
        "--llm_model",
        pins["agent_and_generation_model"],
        "--simulator_model",
        pins["prediction_simulator_model"],
        "--seed",
        str(pins["base_random_seed"]),
        "--num_examples",
        "10",
        "--activation_backend",
        "local" if entry["activation_backend"] == "local_saelens" else "neuronpedia_api",
    ]
    if entry["activation_backend"] == "local_saelens":
        command.extend(
            [
                "--model_revision",
                target["base_model_revision"],
                "--sae_path",
                (
                    "sae-lens://release="
                    f"{target['saelens_release']};sae_id=layer_{entry['layer']}"
                ),
                "--device",
                device,
            ]
        )
    return command


def write_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
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
    target, entries = target_entries(manifest, args.target)
    state_path = ROOT / "reproduction" / f"formal_eval_{args.target.replace('-', '_')}.json"
    state: dict[str, Any] = {
        "target": args.target,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "layers": [],
    }
    write_state(state_path, state)

    for entry in entries:
        command = build_command(
            manifest, target, entry, args.output_root, args.device
        )
        layer_state = {
            "layer": entry["layer"],
            "features": entry["features"],
            "command": command,
            "status": "dry_run" if args.dry_run else "running",
            "started_at_utc": utc_now(),
        }
        state["layers"].append(layer_state)
        write_state(state_path, state)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        generation_dir = (
            args.output_root
            / manifest["replication_pins"]["agent_and_generation_model"]
            / target["target_llm"].replace("/", "_")
            / f"layer_{entry['layer']}"
        )
        missing = [
            feature
            for feature in entry["features"]
            if not generation_result_valid(
                generation_dir / f"feature_{feature}" / "structured_results.json"
            )
        ]
        if missing:
            layer_state["status"] = "blocked_missing_generation"
            layer_state["missing_features"] = missing
            state["status"] = "blocked_missing_generation"
            write_state(state_path, state)
            print(f"missing generation results for layer {entry['layer']}: {missing}")
            return 2

        completed = subprocess.run(command, cwd=ROOT, check=False)
        layer_state["returncode"] = completed.returncode
        layer_state["finished_at_utc"] = utc_now()
        layer_state["status"] = "completed" if completed.returncode == 0 else "failed"
        write_state(state_path, state)
        if completed.returncode != 0:
            state["status"] = "failed"
            state["failed_layer"] = entry["layer"]
            state["finished_at_utc"] = utc_now()
            write_state(state_path, state)
            return completed.returncode

    state["status"] = "dry_run" if args.dry_run else "completed"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
