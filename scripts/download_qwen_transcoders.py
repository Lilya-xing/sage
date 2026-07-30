#!/usr/bin/env python3
"""Download the four formal Qwen transcoder layers concurrently and verify them."""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.hf_download_auth import load_hf_token, run_aria2


REVISION = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
EXPECTED_SIZE = 1_678_054_736
EXPECTED_SHA256 = {
    3: "cc1502ab55af99900d5f754f9fed398401459271056e83ea69a1f8dc5b5c8f1e",
    7: "0c5ae762ac790899ab00890a5cbddf363f375e44597ddfccf69f71c70ca8b217",
    11: "36d7a41b44250c0f02c4bc983d5f2f451e1a4a223facef1a34a8dcaeb3245aad",
    23: "68ab814243fdd7d505604c316799ff5a8791544ed3d0cc5c52d5bd1df0ac01ac",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_layer(
    layer: int, assets_dir: Path, token: str | None, connections: int
) -> Path:
    path = assets_dir / f"layer_{layer}.safetensors"
    control_path = Path(f"{path}.aria2")
    expected_sha256 = EXPECTED_SHA256[layer]
    url = (
        "https://huggingface.co/mwhanna/qwen3-4b-transcoders/resolve/"
        f"{REVISION}/layer_{layer}.safetensors?download=true"
    )

    while (
        not path.exists()
        or path.stat().st_size != EXPECTED_SIZE
        or control_path.exists()
    ):
        print(f"[layer {layer}] starting/resuming download", flush=True)
        run_aria2(
            [
                "-c",
                f"-x{connections}",
                f"-s{connections}",
                "-k4M",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=5",
                "--retry-wait=10",
                "--timeout=120",
                "--connect-timeout=30",
                "--lowest-speed-limit=1K",
                f"--dir={assets_dir}",
                f"--out={path.name}",
                url,
            ],
            token,
        )
        if control_path.exists() or not path.exists() or path.stat().st_size != EXPECTED_SIZE:
            time.sleep(10)

    if control_path.exists():
        raise RuntimeError(f"aria2 control file remains after download: {control_path}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for layer {layer}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    print(f"[layer {layer}] verified {actual_sha256}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--connections", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()
    print(
        "Hugging Face download authentication: "
        f"{'enabled' if token else 'disabled'}",
        flush=True,
    )
    layers = sorted(EXPECTED_SHA256)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                download_layer, layer, args.assets_dir, token, args.connections
            ): layer
            for layer in layers
        }
        for future in as_completed(futures):
            future.result()
    print("All formal Qwen transcoders downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
