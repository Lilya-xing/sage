"""Run the frozen Qwen matrix by layer on a pool of physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def result_has_valid_conclusion(path: Path) -> bool:
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


def missing_features(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> list[int]:
    model_dir = next(
        target["target_llm"].replace("/", "_")
        for target in manifest["targets"]
        if target["neuronpedia_model_id"] == "qwen3-4b"
    )
    layer_dir = (
        output_root
        / manifest["replication_pins"]["agent_and_generation_model"]
        / model_dir
        / f"layer_{entry['layer']}"
    )
    return [
        feature
        for feature in entry["features"]
        if not result_has_valid_conclusion(
            layer_dir / f"feature_{feature}" / "structured_results.json"
        )
    ]


def build_layer_command(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    features: Sequence[int],
    output_root: Path,
) -> list[str]:
    target = next(
        target
        for target in manifest["targets"]
        if target["neuronpedia_model_id"] == "qwen3-4b"
    )
    spec = manifest["paper_main_spec"]
    pins = manifest["replication_pins"]
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
        f"layer{entry['layer']}={','.join(map(str, features))}",
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
        "cuda:0",
    ]


def run_layer(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    physical_gpu: str,
    max_attempts: int,
) -> dict[str, Any]:
    layer = entry["layer"]
    layer_state_path = (
        ROOT / "reproduction" / f"formal_runner_qwen3_4b_layer_{layer}.json"
    )
    log_path = (
        output_root / "_logs" / "aggregate" / f"qwen3-4b_layer{layer}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "target": "qwen3-4b",
        "layer": layer,
        "physical_gpu": physical_gpu,
        "status": "running",
        "started_at_utc": utc_now(),
        "attempts": [],
    }
    write_state(layer_state_path, state)

    for attempt in range(1, max_attempts + 1):
        missing = missing_features(manifest, entry, output_root)
        if not missing:
            state["status"] = "completed"
            state["finished_at_utc"] = utc_now()
            write_state(layer_state_path, state)
            return state

        command = build_layer_command(manifest, entry, missing, output_root)
        attempt_state = {
            "attempt": attempt,
            "features": missing,
            "command": command,
            "started_at_utc": utc_now(),
        }
        state["attempts"].append(attempt_state)
        state["missing_features"] = missing
        write_state(layer_state_path, state)

        env = os.environ.copy()
        hf_home = env.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        hf_hub_cache = env.get(
            "HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub")
        )
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": hf_home,
                "HUGGINGFACE_HUB_CACHE": hf_hub_cache,
                "CUDA_VISIBLE_DEVICES": physical_gpu,
            }
        )
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n[{utc_now()}] layer={layer} attempt={attempt} "
                f"physical_gpu={physical_gpu} features={missing}\n"
            )
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        remaining = missing_features(manifest, entry, output_root)
        attempt_state["returncode"] = completed.returncode
        attempt_state["finished_at_utc"] = utc_now()
        attempt_state["remaining_features"] = remaining
        state["missing_features"] = remaining
        write_state(layer_state_path, state)
        if not remaining:
            state["status"] = "completed"
            state["finished_at_utc"] = utc_now()
            write_state(layer_state_path, state)
            return state

    state["status"] = "blocked_missing_generation"
    state["finished_at_utc"] = utc_now()
    write_state(layer_state_path, state)
    return state


def parse_devices(raw: str) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("devices must be a non-empty unique CSV")
    return devices


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
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("4,6,7"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = qwen_entries(manifest)
    state_path = ROOT / "reproduction" / "formal_runner_qwen3_4b.json"
    state: dict[str, Any] = {
        "target": "qwen3-4b",
        "mode": "multi_gpu_layer_queue",
        "manifest": str(args.manifest),
        "physical_gpus": args.devices,
        "max_attempts": args.max_attempts,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "layers": {},
    }
    write_state(state_path, state)
    if args.dry_run:
        for entry in entries:
            state["layers"][str(entry["layer"])] = {
                "features": entry["features"],
                "command": build_layer_command(
                    manifest, entry, entry["features"], args.output_root
                ),
            }
        write_state(state_path, state)
        return 0

    available_devices: queue.Queue[str] = queue.Queue()
    for device in args.devices:
        available_devices.put(device)

    def run_on_available_device(entry: dict[str, Any]) -> dict[str, Any]:
        device = available_devices.get()
        try:
            return run_layer(
                manifest,
                entry,
                args.output_root,
                device,
                args.max_attempts,
            )
        finally:
            available_devices.put(device)

    failed = False
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = {pool.submit(run_on_available_device, entry): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            result = future.result()
            state["layers"][str(entry["layer"])] = result
            failed = failed or result["status"] != "completed"
            write_state(state_path, state)

    state["status"] = "blocked_missing_generation" if failed else "completed"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
