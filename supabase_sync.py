"""Supabase Storage sync — used by both Windows agent and cloud viewer."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DEFAULT_BUCKET = "usage-tracker"


def upload_files(client, bucket: str, files: Iterable[Path]) -> None:
    """Upload each local file to the bucket, overwriting if it exists."""
    for f in files:
        data = Path(f).read_bytes()
        client.storage.from_(bucket).upload(
            path=Path(f).name,
            file=data,
            file_options={"upsert": "true", "content-type": "application/octet-stream"},
        )


def download_files(
    client, bucket: str, names: Iterable[str], target_dir: Path, prefix: str = ""
) -> None:
    """Download each `name` to `target_dir/name`. When `prefix` is set, the
    remote object key is `prefix/name` (the per-user folder the Rust agent writes
    to); the local filename stays bare so existing config paths still resolve."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        remote = f"{prefix}/{name}" if prefix else name
        data = client.storage.from_(bucket).download(remote)
        (target_dir / name).write_bytes(data)


def last_modified_at(client, bucket: str, name: str, prefix: str = "") -> datetime | None:
    items = client.storage.from_(bucket).list(path=prefix)
    for it in items:
        if it.get("name") == name:
            raw = it.get("updated_at")
            if not raw:
                return None
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
    return None


def from_env():
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    bucket = os.environ.get("SUPABASE_BUCKET", DEFAULT_BUCKET)
    return create_client(url, key), bucket


def from_streamlit_secrets():
    import streamlit as st
    from supabase import create_client
    s = st.secrets["supabase"]
    return create_client(s["url"], s["anon_key"]), s.get("bucket", DEFAULT_BUCKET)
