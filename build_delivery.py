"""Build the source-only research handoff manifest and ZIP."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = ROOT.parent / f"{ROOT.name}_source.zip"
MANIFEST_PATH = ROOT / "DELIVERY_MANIFEST.json"
EXCLUDED_DIRECTORY_NAMES = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "calibrated", "checkpoints", "htmlcov", "prepared",
    "result", "results", "runs", "superpowers",
}
EXCLUDED_SUFFIXES = {".coverage", ".pyc", ".pyo", ".pt", ".pth", ".tmp"}
TEXT_FILE_NAMES = {".gitattributes", ".gitignore"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def _included_files():
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path == MANIFEST_PATH:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def _delivery_bytes(path):
    content = path.read_bytes()
    if path.name in TEXT_FILE_NAMES or path.suffix.lower() in TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def _sha256(path):
    return hashlib.sha256(_delivery_bytes(path)).hexdigest()


def build_delivery(manifest_only=False):
    files = _included_files()
    manifest = {
        "schema_version": 2,
        "package": ROOT.name,
        "kind": "source_only_research_handoff",
        "content_hash_contract": {
            "algorithm": "sha256",
            "binary_content": "raw_bytes",
            "text_content": "crlf_and_cr_normalized_to_lf",
            "text_file_names": sorted(TEXT_FILE_NAMES),
            "text_suffixes": sorted(TEXT_SUFFIXES),
        },
        "full_training_performed_during_preparation": False,
        "excluded": sorted(EXCLUDED_DIRECTORY_NAMES),
        "manifest_self_hash_included": False,
        "file_count_excluding_manifest": len(files),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(_delivery_bytes(path)),
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    if manifest_only:
        return MANIFEST_PATH, None

    temporary = ARCHIVE_PATH.with_name(f".{ARCHIVE_PATH.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            prefix = ROOT.name
            for path in files:
                archive.writestr(
                    f"{prefix}/{path.relative_to(ROOT).as_posix()}",
                    _delivery_bytes(path),
                )
            archive.write(MANIFEST_PATH, f"{prefix}/{MANIFEST_PATH.name}")
        os.replace(temporary, ARCHIVE_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    return MANIFEST_PATH, ARCHIVE_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="refresh repository hashes without creating a ZIP archive",
    )
    args = parser.parse_args()
    manifest_path, archive_path = build_delivery(
        manifest_only=args.manifest_only
    )
    print(f"Manifest: {manifest_path}")
    if archive_path is not None:
        print(f"Archive: {archive_path}")
