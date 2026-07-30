"""Fail-fast environment checks for SAGE reproduction modes."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Iterable


MODULES = {
    "openai": "openai",
    "requests": "requests",
    "numpy": "numpy",
    "scipy": "scipy",
    "python-dotenv": "dotenv",
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sae-lens": "sae_lens",
    "transformer-lens": "transformer_lens",
}


def missing_modules(packages: Iterable[str]) -> list[str]:
    return [
        package
        for package in packages
        if importlib.util.find_spec(MODULES[package]) is None
    ]


def packages_for_mode(mode: str) -> list[str]:
    packages = ["openai", "requests", "python-dotenv"]
    if mode == "paper":
        packages.extend(["numpy", "scipy"])
    elif mode == "local":
        packages.extend(
            [
                "numpy",
                "scipy",
                "torch",
                "transformers",
                "datasets",
                "sae-lens",
                "transformer-lens",
            ]
        )
    return packages


def load_project_env() -> None:
    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / "sage_config.env")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("api", "local", "paper"),
        default="api",
        help="api: one SAGE run; local: target model + SAE; paper: metrics/batch reproduction",
    )
    args = parser.parse_args()

    load_project_env()

    packages = packages_for_mode(args.mode)
    keys = ["OPENAI_API_KEY"]

    missing = missing_modules(packages)
    missing_keys = [key for key in keys if not os.getenv(key)]

    print(f"mode={args.mode}")
    for package in packages:
        status = "missing" if package in missing else "present"
        print(f"package:{package}={status}")
    for key in keys:
        status = "missing" if key in missing_keys else "present"
        print(f"credential:{key}={status}")

    if missing or missing_keys:
        print("preflight=BLOCKED")
        return 2

    print("preflight=READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

