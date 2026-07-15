from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shared import (  # noqa: E402
    DataContractError,
    data_entries,
    file_sha256,
    load_data_manifest,
    validate_image_corpus,
)


def selected_entries(manifest: dict[str, Any], datasets: Iterable[str]) -> list[dict[str, Any]]:
    selected = set(datasets)
    return [entry for entry in manifest["files"] if entry["dataset"] in selected]


def preflight_targets(root: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for entry in entries:
        target = root / entry["local_path"]
        if target.is_symlink():
            raise DataContractError(f"Refusing to overwrite symlink: {target}")
        if not target.exists():
            missing.append(entry)
            continue
        if not target.is_file():
            raise DataContractError(f"Refusing to overwrite non-file: {target}")
        if target.stat().st_size != entry["size"] or file_sha256(target) != entry["sha256"]:
            raise DataContractError(f"Refusing to overwrite unexpected file: {target}")
    return missing


def safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DataContractError(f"Unsafe archive path: {name!r}")
    return path


def restore_from_archive(
    archive: Path,
    root: Path,
    manifest: dict[str, Any],
    datasets: Iterable[str],
) -> None:
    entries = selected_entries(manifest, datasets)
    missing = preflight_targets(root, entries)
    if not missing:
        return

    expected = {
        PurePosixPath(manifest["source"]["archive_root"]) / entry["source_path"]: entry
        for entry in entries
    }
    found: set[PurePosixPath] = set()
    with tempfile.TemporaryDirectory(prefix="rbt4dnn-restore-") as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                member_path = safe_archive_name(member.name)
                if member.issym() or member.islnk():
                    raise DataContractError(f"Archive contains a link: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise DataContractError(f"Archive contains a special entry: {member.name}")
                entry = expected.get(member_path)
                if entry is None:
                    continue
                if not member.isfile():
                    raise DataContractError(f"Expected file is not regular: {member.name}")
                source = tar.extractfile(member)
                if source is None:
                    raise DataContractError(f"Could not read archive member: {member.name}")
                staged = staging / entry["local_path"]
                staged.parent.mkdir(parents=True, exist_ok=True)
                with source, staged.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                found.add(member_path)

        absent = set(expected).difference(found)
        if absent:
            first = min(str(path) for path in absent)
            raise DataContractError(
                f"Pinned archive is missing {len(absent)} allowlisted files (first: {first})"
            )
        for entry in entries:
            staged = staging / entry["local_path"]
            if staged.stat().st_size != entry["size"] or file_sha256(staged) != entry["sha256"]:
                raise DataContractError(f"Checksum mismatch in archive: {entry['source_path']}")

        # Recheck immediately before copying so an existing unexpected file is never replaced.
        missing_paths = {entry["local_path"] for entry in preflight_targets(root, entries)}
        for entry in entries:
            if entry["local_path"] not in missing_paths:
                continue
            target = root / entry["local_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / entry["local_path"], target)


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "rbt4dnn-seminar-fetch/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the exact ignored image corpus from the pinned upstream artifact."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["celeba-hq", "imagenet", "mnist", "sgsm"],
        help="Restore or verify only this dataset (repeatable).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing files without downloading or writing anything.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded pinned GitHub source archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_data_manifest(ROOT)
    datasets = args.dataset or list(manifest["datasets"])
    # Validate selection through the shared contract before any network or filesystem write.
    data_entries(ROOT, datasets)
    if args.verify:
        validate_image_corpus(ROOT, datasets)
        print(f"Verified {len(selected_entries(manifest, datasets))} files.")
        return

    entries = selected_entries(manifest, datasets)
    missing = preflight_targets(ROOT, entries)
    if not missing:
        validate_image_corpus(ROOT, datasets)
        print(f"Already complete: {len(entries)} files verified.")
        return

    if args.archive:
        restore_from_archive(args.archive, ROOT, manifest, datasets)
    else:
        with tempfile.TemporaryDirectory(prefix="rbt4dnn-download-") as temporary:
            archive = Path(temporary) / "RBT4DNN.tar.gz"
            print(
                f"Downloading pinned upstream archive for {len(missing)} missing files...",
                flush=True,
            )
            download(manifest["source"]["archive_url"], archive)
            restore_from_archive(archive, ROOT, manifest, datasets)
    validate_image_corpus(ROOT, datasets)
    print(f"Restored and verified {len(entries)} files.")


if __name__ == "__main__":
    main()
