"""Download and verify pinned Qwen assets with Hugging Face's Xet backend."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

from scripts.download_qwen_base import install_file
from scripts.hf_download_auth import load_hf_token


@dataclass(frozen=True)
class Asset:
    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str
    target: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_verified(asset: Asset) -> bool:
    control = Path(f"{asset.target}.aria2")
    return (
        asset.target.is_file()
        and not control.exists()
        and asset.target.stat().st_size == asset.size
        and file_sha256(asset.target) == asset.sha256
    )


def download(asset: Asset, cache_dir: Path, token: str) -> tuple[Asset, Path]:
    if is_verified(asset):
        print(f"already_verified={asset.target}", flush=True)
        return asset, asset.target
    print(f"xet_download_start={asset.repo_id}:{asset.filename}", flush=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.filename,
            revision=asset.revision,
            cache_dir=cache_dir,
            token=token,
        )
    ).resolve()
    if downloaded.stat().st_size != asset.size:
        raise ValueError(
            f"wrong size for {asset.filename}: {downloaded.stat().st_size} != {asset.size}"
        )
    actual = file_sha256(downloaded)
    if actual != asset.sha256:
        raise ValueError(
            f"checksum mismatch for {asset.filename}: {actual} != {asset.sha256}"
        )
    print(f"xet_download_verified={asset.filename}:{actual}", flush=True)
    return asset, downloaded


def publish(asset: Asset, source: Path, backup_dir: Path) -> None:
    if source == asset.target:
        return
    asset.target.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for old_path in (asset.target, Path(f"{asset.target}.aria2")):
        if old_path.exists() or old_path.is_symlink():
            backup_target = backup_dir / old_path.name
            if backup_target.exists():
                raise FileExistsError(backup_target)
            old_path.rename(backup_target)
            print(f"partial_backup={old_path}->{backup_target}", flush=True)
    try:
        os.link(source, asset.target)
    except OSError:
        shutil.copy2(source, asset.target)
    print(f"published={asset.target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xet-cache", type=Path, required=True)
    parser.add_argument("--base-assets-dir", type=Path, required=True)
    parser.add_argument("--transcoder-assets-dir", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()

    token = load_hf_token()
    if token is None:
        raise RuntimeError("HF_TOKEN is required for the Xet accelerator")
    if args.parallel < 1:
        raise ValueError("parallel must be positive")

    base_revision = "1cfa9a7208912126459214e8b04321603b3df60c"
    trans_revision = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
    base_specs = {
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
    trans_specs = {
        3: "cc1502ab55af99900d5f754f9fed398401459271056e83ea69a1f8dc5b5c8f1e",
        7: "0c5ae762ac790899ab00890a5cbddf363f375e44597ddfccf69f71c70ca8b217",
        11: "36d7a41b44250c0f02c4bc983d5f2f451e1a4a223facef1a34a8dcaeb3245aad",
        23: "68ab814243fdd7d505604c316799ff5a8791544ed3d0cc5c52d5bd1df0ac01ac",
    }
    assets = [
        Asset(
            "Qwen/Qwen3-4B",
            base_revision,
            filename,
            size,
            sha256,
            args.base_assets_dir / filename,
        )
        for filename, (size, sha256) in base_specs.items()
    ]
    assets.extend(
        Asset(
            "mwhanna/qwen3-4b-transcoders",
            trans_revision,
            f"layer_{layer}.safetensors",
            1_678_054_736,
            sha256,
            args.transcoder_assets_dir / f"layer_{layer}.safetensors",
        )
        for layer, sha256 in trans_specs.items()
    )

    args.xet_cache.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.backup_root / f"aria2-partials-{timestamp}"
    downloaded: list[tuple[Asset, Path]] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(download, asset, args.xet_cache, token): asset
            for asset in assets
        }
        for future in as_completed(futures):
            downloaded.append(future.result())

    for asset, source in downloaded:
        publish(asset, source, backup_dir)
    for asset in assets:
        if asset.repo_id == "Qwen/Qwen3-4B":
            install_file(asset.target, args.hub_cache)
    print(f"all_xet_assets_verified=7 backup_dir={backup_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
