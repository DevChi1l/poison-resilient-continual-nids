import zipfile
from pathlib import Path

from scripts.prepare_raw_data import ExtractionError, prepare_raw_data


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, content in members.items():
            archive.writestr(member_name, content)


def test_extracts_multiple_archives_without_changing_sources(tmp_path):
    source_dir = tmp_path / "source"
    raw_dir = tmp_path / "raw"
    source_dir.mkdir()
    first_archive = source_dir / "first.zip"
    second_archive = source_dir / "second.zip"
    _write_zip(first_archive, {"first.txt": "first"})
    _write_zip(second_archive, {"nested/second.txt": "second"})
    original_first_archive = first_archive.read_bytes()

    first_run = prepare_raw_data(source_dir, raw_dir)
    assert first_run.archives == 2
    assert first_run.files_extracted == 2
    assert first_run.files_skipped == 0
    assert (raw_dir / "first.txt").read_text() == "first"
    assert (raw_dir / "nested" / "second.txt").read_text() == "second"
    assert first_archive.read_bytes() == original_first_archive

    extracted_mtime = (raw_dir / "first.txt").stat().st_mtime_ns
    second_run = prepare_raw_data(source_dir, raw_dir)
    assert second_run.files_extracted == 0
    assert second_run.files_skipped == 2
    assert (raw_dir / "first.txt").stat().st_mtime_ns == extracted_mtime


def test_rejects_archive_paths_that_escape_raw_directory(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_zip(source_dir / "unsafe.zip", {"../escape.txt": "blocked"})

    try:
        prepare_raw_data(source_dir, tmp_path / "raw")
    except ExtractionError as error:
        assert "Unsafe ZIP member path" in str(error)
    else:
        raise AssertionError("Expected unsafe archive path to be rejected")


def test_dry_run_reports_work_without_creating_raw_directory(tmp_path):
    source_dir = tmp_path / "source"
    raw_dir = tmp_path / "raw"
    source_dir.mkdir()
    _write_zip(source_dir / "dataset.zip", {"flow.parquet": "not-real-parquet"})

    summary = prepare_raw_data(source_dir, raw_dir, dry_run=True)

    assert summary.archives == 1
    assert summary.files_extracted == 1
    assert summary.files_skipped == 0
    assert summary.dry_run is True
    assert not raw_dir.exists()
