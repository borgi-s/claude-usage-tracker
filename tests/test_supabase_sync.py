"""Unit tests for supabase_sync — mocked Supabase client."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import supabase_sync


@pytest.fixture
def fake_client():
    client = MagicMock()
    client.storage.from_.return_value = client.storage
    client.storage.upload = MagicMock(return_value={"path": "ok"})
    client.storage.download = MagicMock(return_value=b"binary content")
    client.storage.list = MagicMock(return_value=[
        {"name": "cache.parquet", "updated_at": "2026-05-21T12:00:00Z"},
    ])
    return client


def test_upload_files_calls_upload_for_each_file(tmp_path: Path, fake_client):
    f1 = tmp_path / "cache.parquet"; f1.write_bytes(b"abc")
    f2 = tmp_path / "caps.json"; f2.write_bytes(b"def")
    supabase_sync.upload_files(fake_client, "usage-tracker", [f1, f2])
    assert fake_client.storage.upload.call_count == 2


def test_download_files_writes_each_to_target_dir(tmp_path: Path, fake_client):
    supabase_sync.download_files(
        fake_client, "usage-tracker",
        ["cache.parquet", "caps.json"],
        target_dir=tmp_path,
    )
    assert (tmp_path / "cache.parquet").read_bytes() == b"binary content"
    assert (tmp_path / "caps.json").read_bytes() == b"binary content"
    assert fake_client.storage.download.call_count == 2


def test_last_modified_at_returns_parsed_datetime(fake_client):
    ts = supabase_sync.last_modified_at(fake_client, "usage-tracker", "cache.parquet")
    assert ts is not None
    assert ts.year == 2026 and ts.month == 5 and ts.day == 21


def test_download_cache_per_prefix_writes_distinct_local_files(tmp_path, fake_client):
    result = supabase_sync.download_cache_per_prefix(
        fake_client, "usage-tracker", ["borgi", "borgi-linux"], target_dir=tmp_path
    )
    assert set(result.keys()) == {"borgi", "borgi-linux"}
    # distinct local filenames per prefix (no collision)
    assert result["borgi"] != result["borgi-linux"]
    for p in result.values():
        assert p.exists()
        assert p.read_bytes() == b"binary content"
    # downloaded cache.parquet under each prefix
    downloaded = [c.args[0] for c in fake_client.storage.download.call_args_list]
    assert "borgi/cache.parquet" in downloaded
    assert "borgi-linux/cache.parquet" in downloaded


def test_download_cache_per_prefix_skips_missing_object(tmp_path, fake_client):
    # First prefix downloads fine; second raises (no object yet) and is skipped.
    def _dl(remote):
        if remote.startswith("borgi-linux/"):
            raise Exception("Object not found")
        return b"binary content"
    fake_client.storage.download = MagicMock(side_effect=_dl)

    result = supabase_sync.download_cache_per_prefix(
        fake_client, "usage-tracker", ["borgi", "borgi-linux"], target_dir=tmp_path
    )
    assert set(result.keys()) == {"borgi"}
    assert result["borgi"].exists()
