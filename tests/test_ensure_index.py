"""ensure_index() is the guard app.py's lifespan calls on every cold boot
(Render included): it must skip a full re-ingest when the corpus is already
built, and must build it exactly once when it isn't. Get either branch wrong
and either every deploy re-embeds the whole corpus, or a fresh container
serves an empty index.
"""
import shutil
import tarfile
from pathlib import Path
from unittest.mock import Mock

from src import config, ingestion


def test_ensure_index_skips_ingest_when_marker_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path)
    (tmp_path / ingestion._MARKER_NAME).touch()

    spy = Mock()
    monkeypatch.setattr(ingestion, "ingest", spy)

    ingestion.ensure_index()

    spy.assert_not_called()


def test_ensure_index_calls_ingest_once_when_marker_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path)
    assert not (tmp_path / ingestion._MARKER_NAME).exists()

    spy = Mock()
    monkeypatch.setattr(ingestion, "ingest", spy)

    ingestion.ensure_index()

    spy.assert_called_once_with(rebuild=True)


def _index_tarball(tmp_path: Path) -> Path:
    built = tmp_path / "build" / ".chroma"
    built.mkdir(parents=True)
    (built / ingestion._MARKER_NAME).touch()
    (built / "chroma.sqlite3").write_text("stub")
    archive = tmp_path / "chroma.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(built, arcname=".chroma")
    return archive


def test_ensure_index_restores_from_s3_without_ingesting(tmp_path, monkeypatch):
    archive = _index_tarball(tmp_path)
    chroma = tmp_path / "state" / ".chroma"
    monkeypatch.setattr(config, "CHROMA_DIR", chroma)
    monkeypatch.setattr(config, "INDEX_S3_URI", "s3://bucket/index/chroma.tgz")
    monkeypatch.setattr(ingestion, "_download", lambda uri, dest: shutil.copy(archive, dest))
    spy = Mock()
    monkeypatch.setattr(ingestion, "ingest", spy)

    ingestion.ensure_index()

    spy.assert_not_called()
    assert (chroma / ingestion._MARKER_NAME).exists()


def test_ensure_index_rejects_archive_without_marker(tmp_path, monkeypatch):
    built = tmp_path / "build" / ".chroma"
    built.mkdir(parents=True)
    (built / "chroma.sqlite3").write_text("half-built")
    archive = tmp_path / "partial.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(built, arcname=".chroma")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "state" / ".chroma")
    monkeypatch.setattr(config, "INDEX_S3_URI", "s3://bucket/index/partial.tgz")
    monkeypatch.setattr(ingestion, "_download", lambda uri, dest: shutil.copy(archive, dest))
    spy = Mock()
    monkeypatch.setattr(ingestion, "ingest", spy)

    ingestion.ensure_index()

    spy.assert_called_once_with(rebuild=True)


def test_ensure_index_falls_back_to_ingest_when_restore_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "state" / ".chroma")
    monkeypatch.setattr(config, "INDEX_S3_URI", "s3://bucket/index/missing.tgz")

    def boom(uri, dest):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(ingestion, "_download", boom)
    spy = Mock()
    monkeypatch.setattr(ingestion, "ingest", spy)

    ingestion.ensure_index()

    spy.assert_called_once_with(rebuild=True)
