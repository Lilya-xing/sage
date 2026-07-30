"""Verify pinned Qwen transcoders and expose them through the HF cache."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


REVISION = "94d176260ac39ce2f882b8b09aba8c118df29bb3"
EXPECTED = {
    3: "cc1502ab55af99900d5f754f9fed398401459271056e83ea69a1f8dc5b5c8f1e",
    7: "0c5ae762ac790899ab00890a5cbddf363f375e44597ddfccf69f71c70ca8b217",
    11: "36d7a41b44250c0f02c4bc983d5f2f451e1a4a223facef1a34a8dcaeb3245aad",
    23: "68ab814243fdd7d505604c316799ff5a8791544ed3d0cc5c52d5bd1df0ac01ac",
}
EXPECTED_SIZE = 1_678_054_736


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_cache = args.cache_dir / "models--mwhanna--qwen3-4b-transcoders"
    blobs = repo_cache / "blobs"
    snapshot = repo_cache / "snapshots" / REVISION
    refs = repo_cache / "refs"
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)

    for layer, expected_sha in EXPECTED.items():
        source = args.assets_dir / f"layer_{layer}.safetensors"
        if source.stat().st_size != EXPECTED_SIZE:
            raise ValueError(f"wrong size for {source}")
        actual_sha = file_sha256(source)
        if actual_sha != expected_sha:
            raise ValueError(
                f"checksum mismatch for {source}: {actual_sha} != {expected_sha}"
            )

        blob = blobs / expected_sha
        if blob.exists():
            if blob.stat().st_size != EXPECTED_SIZE or file_sha256(blob) != expected_sha:
                raise ValueError(f"existing cache blob is invalid: {blob}")
        else:
            os.link(source, blob)

        link = snapshot / f"layer_{layer}.safetensors"
        expected_target = Path("../../blobs") / expected_sha
        if link.is_symlink():
            if Path(os.readlink(link)) != expected_target:
                raise ValueError(f"unexpected existing symlink: {link}")
        elif link.exists():
            raise ValueError(f"unexpected existing snapshot file: {link}")
        else:
            link.symlink_to(expected_target)
        print(f"verified_and_linked=layer_{layer}:{expected_sha}")

    main_ref = refs / "main"
    if main_ref.exists() and main_ref.read_text(encoding="utf-8").strip() != REVISION:
        raise ValueError(f"ref main does not match pinned revision: {main_ref}")
    # huggingface_hub compares this value byte-for-byte with snapshot names.
    main_ref.write_text(REVISION, encoding="utf-8")
    print(f"revision={REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
