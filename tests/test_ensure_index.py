"""ensure_index() is the guard app.py's lifespan calls on every cold boot
(Render included): it must skip a full re-ingest when the corpus is already
built, and must build it exactly once when it isn't. Get either branch wrong
and either every deploy re-embeds the whole corpus, or a fresh container
serves an empty index.
"""
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
