"""Resume the pinned Qwen base shards with aria2 and install verified cache links."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.hf_download_auth import load_hf_token, run_aria2


REPO_CACHE_NAME = "models--Qwen--Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
FILES = {
    "model-00001-of-00003.safetensors": (
        3_957_900_840,
        "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    ),
    "model-00002-of-00003.safetensors": (
        3_987_450_520,
        "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    ),
    "model-00003-of-00003.safetensors": (
        99_630_640,
        "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_from_hub_cache(target: Path, hub_cache: Path, sha256: str) -> None:
    if target.exists():
        return
    blobs = hub_cache / REPO_CACHE_NAME / "blobs"
    final_blob = blobs / sha256
    if final_blob.exists():
        os.link(final_blob, target)
        print(f"[{target.name}] seeded from verified blob", flush=True)
        return
    candidates = sorted(blobs.glob(f"{sha256}.*.incomplete"))
    if len(candidates) == 1:
        os.link(candidates[0], target)
        print(
            f"[{target.name}] reused {candidates[0].stat().st_size} cached bytes",
            flush=True,
        )
    elif candidates:
        raise RuntimeError(f"ambiguous incomplete cache files for {target.name}")


def download_file(
    filename: str,
    assets_dir: Path,
    hub_cache: Path,
    connections: int,
    token: str | None,
) -> Path:
    expected_size, expected_sha256 = FILES[filename]
    path = assets_dir / filename
    seed_from_hub_cache(path, hub_cache, expected_sha256)
    control_path = Path(f"{path}.aria2")
    url = (
        "https://huggingface.co/Qwen/Qwen3-4B/resolve/"
        f"{REVISION}/{filename}?download=true"
    )
    while (
        not path.exists()
        or path.stat().st_size != expected_size
        or control_path.exists()
    ):
        print(f"[{filename}] starting/resuming download", flush=True)
        run_aria2(
            [
                "-c",
                f"-x{connections}",
                f"-s{connections}",
                "-k1M",
                "--file-allocation=none",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-tries=0",
                "--retry-wait=3",
                "--timeout=120",
                "--connect-timeout=30",
                "--lowest-speed-limit=1K",
                f"--dir={assets_dir}",
                f"--out={filename}",
                url,
            ],
            token,
        )
        if control_path.exists() or not path.exists() or path.stat().st_size != expected_size:
            time.sleep(10)

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {filename}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    print(f"[{filename}] verified {actual_sha256}", flush=True)
    return path


def install_file(path: Path, hub_cache: Path) -> None:
    _, sha256 = FILES[path.name]
    repo_cache = hub_cache / REPO_CACHE_NAME
    blobs = repo_cache / "blobs"
    snapshot = repo_cache / "snapshots" / REVISION
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    blob = blobs / sha256
    if not blob.exists():
        os.link(path, blob)
    link = snapshot / path.name
    expected_target = Path("../..") / "blobs" / sha256
    if link.is_symlink():
        if Path(os.readlink(link)) != expected_target:
            raise RuntimeError(f"unexpected snapshot link for {path.name}")
    elif link.exists():
        if not os.path.samefile(link, blob):
            raise RuntimeError(f"unexpected snapshot file for {path.name}")
    else:
        link.symlink_to(expected_target)
    print(f"[{path.name}] installed in pinned Hugging Face snapshot", flush=True)


def install_main_ref(hub_cache: Path) -> None:
    repo_cache = hub_cache / REPO_CACHE_NAME
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    main_ref = refs / "main"
    if main_ref.exists() and main_ref.read_text(encoding="utf-8").strip() != REVISION:
        raise RuntimeError(f"ref main does not match pinned revision: {main_ref}")
    # huggingface_hub compares this value byte-for-byte with snapshot names.
    main_ref.write_text(REVISION, encoding="utf-8")
    print(f"[main] installed pinned revision ref {REVISION}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()
    print(
        "Hugging Face download authentication: "
        f"{'enabled' if token else 'disabled'}",
        flush=True,
    )

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                download_file,
                filename,
                args.assets_dir,
                args.hub_cache,
                args.connections,
                token,
            ): filename
            for filename in FILES
        }
        for future in as_completed(futures):
            paths.append(future.result())
    for path in paths:
        install_file(path, args.hub_cache)
    install_main_ref(args.hub_cache)
    print("Pinned Qwen base model downloaded, verified, and installed.", flush=True)


if __name__ == "__main__":
    main()
