"""Run one API-backed target from the frozen formal reproduction manifest.

The runner is intentionally conservative: layers run sequentially, completed
features are left to ``main.py``'s resume guard, and the first failing layer
stops the target. Runtime state is written atomically for monitoring.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("paper_main_spec", {}).get("total_features") != 120:
        raise ValueError("manifest does not contain the paper's 120-feature matrix")
    return manifest


def select_entries(
    manifest: dict[str, Any], target: str, layers: Sequence[int]
) -> list[dict[str, Any]]:
    selected = [
        entry
        for entry in manifest["aggregate_matrix"]
        if entry["neuronpedia_model_id"] == target and entry["layer"] in layers
    ]
    selected.sort(key=lambda entry: layers.index(entry["layer"]))
    missing = [layer for layer in layers if not any(x["layer"] == layer for x in selected)]
    if missing:
        raise ValueError(f"manifest has no entries for {target} layers {missing}")
    if any(entry["activation_backend"] != "neuronpedia_api" for entry in selected):
        raise ValueError(
            f"{target} requires the local SAE runner; this runner is API-only"
        )
    return selected


def build_command(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> list[str]:
    spec = manifest["paper_main_spec"]
    pins = manifest["replication_pins"]
    return [
        sys.executable,
        str(ROOT / "main.py"),
        "--agent_llm",
        pins["agent_and_generation_model"],
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
        "true",
        "--use_saedashboard",
        "false",
        "--device",
        "cpu",
        "--target_llm",
        entry["target_llm"],
        "--sae_path",
        "api-mode",
        "--features",
        f"layer{entry['layer']}={','.join(map(str, entry['features']))}",
        "--neuronpedia_model_id",
        entry["neuronpedia_model_id"],
        "--neuronpedia_source",
        entry["source"],
    ]


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_pid(pid: int, state_path: Path, state: dict[str, Any]) -> None:
    state["status"] = "waiting_for_previous_layer"
    state["wait_pid"] = pid
    state["updated_at_utc"] = utc_now()
    write_state(state_path, state)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(15)


def parse_layers(raw: str) -> list[int]:
    layers = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not layers or len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("layers must be a non-empty unique CSV")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reproduction" / "formal_manifest.json",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("3,7,11,23"))
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "formal-eacl2026"
    )
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-suffix", default="")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    entries = select_entries(manifest, args.target, args.layers)
    if args.state_suffix and not args.state_suffix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("state-suffix must be alphanumeric with optional _ or -")
    state_suffix = f"_{args.state_suffix}" if args.state_suffix else ""
    state_path = (
        ROOT
        / "reproduction"
        / f"formal_runner_{args.target.replace('-', '_')}{state_suffix}.json"
    )
    state: dict[str, Any] = {
        "target": args.target,
        "manifest": str(args.manifest),
        "layers": args.layers,
        "started_at_utc": utc_now(),
        "status": "starting",
        "runs": [],
    }
    write_state(state_path, state)

    if args.wait_for_pid:
        wait_for_pid(args.wait_for_pid, state_path, state)

    for entry in entries:
        command = build_command(manifest, entry, args.output_root)
        run_state = {
            "layer": entry["layer"],
            "features": entry["features"],
            "source": entry["source"],
            "command": command,
            "status": "dry_run" if args.dry_run else "running",
            "started_at_utc": utc_now(),
        }
        state["runs"].append(run_state)
        state["status"] = "dry_run" if args.dry_run else "running"
        state["updated_at_utc"] = utc_now()
        write_state(state_path, state)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=ROOT, check=False)
        run_state["returncode"] = completed.returncode
        run_state["finished_at_utc"] = utc_now()
        run_state["status"] = "completed" if completed.returncode == 0 else "failed"
        state["updated_at_utc"] = utc_now()
        write_state(state_path, state)
        if completed.returncode != 0:
            state["status"] = "failed"
            state["failed_layer"] = entry["layer"]
            state["updated_at_utc"] = utc_now()
            write_state(state_path, state)
            return completed.returncode

    state["status"] = "dry_run" if args.dry_run else "completed"
    state["finished_at_utc"] = utc_now()
    state["updated_at_utc"] = utc_now()
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
