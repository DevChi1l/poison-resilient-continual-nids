#!/usr/bin/env python3
"""Safely extract one or more dataset ZIP files into a raw-data directory.

This utility deliberately stops at extraction. It does not inspect schemas,
rename labels, sample rows, or preprocess files; those steps belong to the data
and continual-learning pipeline.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ExtractionError(RuntimeError):
    """Raised when an archive cannot be extracted safely without overwriting data."""


@dataclass(frozen=True)
class ExtractionSummary:
    archives: int
    files_extracted: int
    files_skipped: int
    directories_created: int
    dry_run: bool


def find_zip_archives(source_dir: Path) -> list[Path]:
    """Find ZIP files recursively, sorted for reproducible extraction order."""

    if not source_dir.is_dir():
        raise ExtractionError(f"Source directory does not exist: {source_dir}")

    archives = sorted(
        (path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".zip"),
        key=lambda path: str(path).lower(),
    )
    if not archives:
        raise ExtractionError(f"No ZIP files found under: {source_dir}")
    return archives


def prepare_raw_data(source_dir: Path, raw_dir: Path, dry_run: bool = False) -> ExtractionSummary:
    """Extract all source ZIPs without modifying archives or replacing raw files.

    Existing files with the expected uncompressed size are skipped. A file with
    a different size, a leftover partial file, a ZIP symlink, or a path that
    escapes the raw directory causes a clear failure instead of overwriting anything.
    """

    source_dir = source_dir.expanduser().resolve()
    raw_dir = raw_dir.expanduser().resolve()
    archives = find_zip_archives(source_dir)

    if not dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)

    files_extracted = 0
    files_skipped = 0
    directories_created = 0
    print(f"Found {len(archives)} ZIP archive(s) in: {source_dir}")
    print(f"Raw-data destination: {raw_dir}")
    if dry_run:
        print("Dry run enabled: no directories or files will be created.")

    for archive_number, archive_path in enumerate(archives, start=1):
        if not zipfile.is_zipfile(archive_path):
            raise ExtractionError(f"Not a valid ZIP archive: {archive_path}")

        print(f"[{archive_number}/{len(archives)}] Processing {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = _safe_destination(raw_dir, member.filename)
                _reject_symlink(member, archive_path)

                if member.is_dir():
                    if not destination.exists():
                        directories_created += 1
                        if not dry_run:
                            destination.mkdir(parents=True, exist_ok=True)
                    continue

                if destination.exists():
                    if destination.is_dir():
                        raise ExtractionError(
                            f"Expected a file but found a directory: {destination}"
                        )
                    if destination.stat().st_size != member.file_size:
                        raise ExtractionError(
                            "Existing raw file has a different size and will not be replaced: "
                            f"{destination}"
                        )
                    files_skipped += 1
                    print(f"  skip existing: {destination.relative_to(raw_dir)}")
                    continue

                partial_path = destination.with_name(f"{destination.name}.part")
                if partial_path.exists():
                    raise ExtractionError(
                        "A partial extraction exists. Inspect or remove it manually before retrying: "
                        f"{partial_path}"
                    )

                files_extracted += 1
                print(f"  extract: {destination.relative_to(raw_dir)} ({member.file_size:,} bytes)")
                if dry_run:
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with archive.open(member) as source, partial_path.open("xb") as destination_file:
                        shutil.copyfileobj(source, destination_file, length=1024 * 1024)
                    os.replace(partial_path, destination)
                except Exception:
                    # Keep a partial file as an explicit signal; never delete or overwrite data silently.
                    raise

    summary = ExtractionSummary(
        archives=len(archives),
        files_extracted=files_extracted,
        files_skipped=files_skipped,
        directories_created=directories_created,
        dry_run=dry_run,
    )
    action = "would be extracted" if dry_run else "extracted"
    print(
        "Complete: "
        f"{summary.files_extracted} file(s) {action}, "
        f"{summary.files_skipped} existing file(s) skipped, "
        f"{summary.directories_created} directory/directories created."
    )
    return summary


def _safe_destination(raw_dir: Path, member_name: str) -> Path:
    """Return a destination only when an archive member stays below raw_dir."""

    destination = (raw_dir / member_name).resolve()
    if not destination.is_relative_to(raw_dir):
        raise ExtractionError(f"Unsafe ZIP member path rejected: {member_name!r}")
    return destination


def _reject_symlink(member: zipfile.ZipInfo, archive_path: Path) -> None:
    unix_mode = member.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ExtractionError(
            f"ZIP symlink member rejected in {archive_path.name}: {member.filename}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely extract dataset ZIP files without modifying the source archives."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing one or more ZIP archives (searched recursively).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Destination for extracted raw files. Existing matching files are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without writing any files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        prepare_raw_data(args.source_dir, args.raw_dir, dry_run=args.dry_run)
    except (ExtractionError, OSError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

