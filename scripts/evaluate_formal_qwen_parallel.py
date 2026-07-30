"""Evaluate the four formal Qwen layers on a pool of physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluate_formal_target import (
    build_command,
    generation_result_valid,
    target_entries,
)


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_devices(raw: str) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("devices must be a non-empty unique CSV")
    return devices


def missing_generation(
    manifest: dict[str, Any],
    target: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> list[int]:
    generation_dir = (
        output_root
        / manifest["replication_pins"]["agent_and_generation_model"]
        / target["target_llm"].replace("/", "_")
        / f"layer_{entry['layer']}"
    )
    return [
        feature
        for feature in entry["features"]
        if not generation_result_valid(
            generation_dir / f"feature_{feature}" / "structured_results.json"
        )
    ]


def run_layer(
    manifest: dict[str, Any],
    target: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    physical_gpu: str,
) -> dict[str, Any]:
    layer = entry["layer"]
    command = build_command(manifest, target, entry, output_root, "cuda:0")
    state_path = ROOT / "reproduction" / f"formal_eval_qwen3_4b_layer_{layer}.json"
    log_path = (
        output_root
        / "_logs"
        / "aggregate"
        / f"qwen3-4b_layer{layer}_evaluation.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "target": "qwen3-4b",
        "layer": layer,
        "physical_gpu": physical_gpu,
        "command": command,
        "status": "running",
        "started_at_utc": utc_now(),
    }
    missing = missing_generation(manifest, target, entry, output_root)
    if missing:
        state["status"] = "blocked_missing_generation"
        state["missing_features"] = missing
        state["finished_at_utc"] = utc_now()
        write_state(state_path, state)
        return state
    write_state(state_path, state)

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
            f"\n[{utc_now()}] layer={layer} physical_gpu={physical_gpu}\n"
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
    state["returncode"] = completed.returncode
    state["status"] = "completed" if completed.returncode == 0 else "failed"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return state


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    target, entries = target_entries(manifest, "qwen3-4b")
    state_path = ROOT / "reproduction" / "formal_eval_qwen3_4b.json"
    state: dict[str, Any] = {
        "target": "qwen3-4b",
        "mode": "multi_gpu_layer_queue",
        "physical_gpus": args.devices,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "layers": {},
    }
    if args.dry_run:
        for entry in entries:
            state["layers"][str(entry["layer"])] = {
                "features": entry["features"],
                "command": build_command(
                    manifest, target, entry, args.output_root, "cuda:0"
                ),
            }
        write_state(state_path, state)
        return 0
    write_state(state_path, state)

    available_devices: queue.Queue[str] = queue.Queue()
    for device in args.devices:
        available_devices.put(device)

    def run_on_available_device(entry: dict[str, Any]) -> dict[str, Any]:
        device = available_devices.get()
        try:
            return run_layer(manifest, target, entry, args.output_root, device)
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

    state["status"] = "failed" if failed else "completed"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
