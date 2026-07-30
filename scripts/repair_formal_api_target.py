"""Audit and retry only missing formal generations for one API-backed target."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluate_formal_target import generation_result_valid
from scripts.run_formal_target import build_command, select_entries


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def missing_features(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
) -> list[int]:
    layer_dir = (
        output_root
        / manifest["replication_pins"]["agent_and_generation_model"]
        / entry["target_llm"].replace("/", "_")
        / f"layer_{entry['layer']}"
    )
    return [
        feature
        for feature in entry["features"]
        if not generation_result_valid(
            layer_dir / f"feature_{feature}" / "structured_results.json"
        )
    ]


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
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = select_entries(manifest, args.target, [3, 7, 11, 23])
    state_path = (
        ROOT / "reproduction" / f"formal_repair_{args.target.replace('-', '_')}.json"
    )
    state: dict[str, Any] = {
        "target": args.target,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "max_attempts": args.max_attempts,
        "attempts": [],
    }
    write_state(state_path, state)

    for attempt in range(1, args.max_attempts + 1):
        attempt_state: dict[str, Any] = {
            "attempt": attempt,
            "started_at_utc": utc_now(),
            "layers": [],
        }
        state["attempts"].append(attempt_state)
        any_missing = False
        for entry in entries:
            missing = missing_features(manifest, entry, args.output_root)
            if not missing:
                continue
            any_missing = True
            retry_entry = dict(entry)
            retry_entry["features"] = missing
            command = build_command(manifest, retry_entry, args.output_root)
            layer_state: dict[str, Any] = {
                "layer": entry["layer"],
                "features": missing,
                "command": command,
                "status": "dry_run" if args.dry_run else "running",
                "started_at_utc": utc_now(),
            }
            attempt_state["layers"].append(layer_state)
            write_state(state_path, state)
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=ROOT, check=False)
            layer_state["returncode"] = completed.returncode
            layer_state["remaining_features"] = missing_features(
                manifest, entry, args.output_root
            )
            layer_state["status"] = (
                "completed" if not layer_state["remaining_features"] else "incomplete"
            )
            layer_state["finished_at_utc"] = utc_now()
            write_state(state_path, state)

        attempt_state["finished_at_utc"] = utc_now()
        if args.dry_run:
            state["status"] = "dry_run"
            write_state(state_path, state)
            return 0
        remaining = {
            str(entry["layer"]): missing_features(manifest, entry, args.output_root)
            for entry in entries
        }
        remaining = {layer: values for layer, values in remaining.items() if values}
        state["remaining_by_layer"] = remaining
        write_state(state_path, state)
        if not remaining:
            state["status"] = "completed"
            state["finished_at_utc"] = utc_now()
            write_state(state_path, state)
            return 0
        if not any_missing:
            break

    state["status"] = "blocked_missing_generation"
    state["finished_at_utc"] = utc_now()
    write_state(state_path, state)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
