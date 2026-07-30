"""Project-scoped Hugging Face authentication for aria2 downloads."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def load_hf_token() -> str | None:
    """Load HF_TOKEN without printing it or placing it on a command line."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        config_file = Path(__file__).resolve().parents[1] / "sage_config.env"
        if config_file.exists():
            token = dotenv_values(config_file).get("HF_TOKEN")
    if not token:
        return None
    token = token.strip()
    if not token.startswith("hf_") or "\n" in token or "\r" in token:
        raise RuntimeError("HF_TOKEN is present but has an invalid format")
    return token


def run_aria2(arguments: list[str], token: str | None) -> None:
    """Supply the Authorization header through an inherited anonymous pipe."""
    if token is None:
        subprocess.run(["aria2c", *arguments], check=False)
        return

    config_fd = os.memfd_create("hf-aria2-auth", flags=0)
    try:
        os.write(
            config_fd,
            f"header=Authorization: Bearer {token}\n".encode("utf-8"),
        )
        os.lseek(config_fd, 0, os.SEEK_SET)
        subprocess.run(
            ["aria2c", f"--conf-path=/proc/self/fd/{config_fd}", *arguments],
            check=False,
            pass_fds=(config_fd,),
        )
    finally:
        os.close(config_fd)
