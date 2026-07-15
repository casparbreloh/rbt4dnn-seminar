from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "llm-validity-audit"))

from fetch_data import restore_from_archive  # noqa: E402
from llm_validity_audit import run_audit  # noqa: E402
from shared import DataContractError, validate_image_corpus  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def one_file_manifest(data: bytes = b"image") -> dict[str, object]:
    return {
        "source": {"archive_root": "upstream-root"},
        "datasets": {"mnist": {"file_count": 1}},
        "files": [
            {
                "dataset": "mnist",
                "local_path": "data/images/mnist/M1/0.png",
                "source_path": "data/images_from_loras/M1/0.png",
                "size": len(data),
                "sha256": sha256(data),
            }
        ],
    }


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


class ArchiveRestoreTests(unittest.TestCase):
    def make_archive(self, path: Path, data: bytes = b"image") -> None:
        with tarfile.open(path, "w:gz") as tar:
            add_bytes(tar, "upstream-root/data/images_from_loras/M1/0.png", data)

    def test_restore_allowlisted_file_and_verify_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            self.make_archive(archive)
            restore_from_archive(archive, root, one_file_manifest(), ["mnist"])
            self.assertEqual((root / "data/images/mnist/M1/0.png").read_bytes(), b"image")

    def test_refuses_to_overwrite_unexpected_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "data/images/mnist/M1/0.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"unexpected")
            archive = root / "source.tar.gz"
            self.make_archive(archive)
            with self.assertRaisesRegex(DataContractError, "Refusing to overwrite"):
                restore_from_archive(archive, root, one_file_manifest(), ["mnist"])
            self.assertEqual(target.read_bytes(), b"unexpected")

    def test_rejects_traversal_before_restoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                add_bytes(tar, "../escape", b"bad")
                add_bytes(tar, "upstream-root/data/images_from_loras/M1/0.png", b"image")
            with self.assertRaisesRegex(DataContractError, "Unsafe archive path"):
                restore_from_archive(archive, root, one_file_manifest(), ["mnist"])
            self.assertFalse((root / "data/images/mnist/M1/0.png").exists())

    def test_rejects_symlink_before_restoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                link = tarfile.TarInfo("upstream-root/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "elsewhere"
                tar.addfile(link)
                add_bytes(tar, "upstream-root/data/images_from_loras/M1/0.png", b"image")
            with self.assertRaisesRegex(DataContractError, "contains a link"):
                restore_from_archive(archive, root, one_file_manifest(), ["mnist"])
            self.assertFalse((root / "data/images/mnist/M1/0.png").exists())

    def test_checksum_failure_does_not_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tar.gz"
            self.make_archive(archive, b"wrong")
            with self.assertRaisesRegex(DataContractError, "Checksum mismatch"):
                restore_from_archive(archive, root, one_file_manifest(), ["mnist"])
            self.assertFalse((root / "data/images/mnist/M1/0.png").exists())


class ExecutionContractTests(unittest.TestCase):
    def write_missing_manifest(self, root: Path) -> None:
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "data/requirements.csv").write_text((ROOT / "data/requirements.csv").read_text())
        records = []
        for dataset, requirement in [("mnist", "M1"), ("celeba-hq", "C1"), ("sgsm", "S1")]:
            records.append(
                "\t".join(
                    [
                        "0" * 64,
                        "1",
                        dataset,
                        f"data/images/{dataset}/{requirement}/0.png",
                        f"data/images_from_loras/{requirement}/0.png",
                    ]
                )
            )
        (root / "data/manifest-files.tsv").write_text("\n".join(records) + "\n")
        (root / "data/manifest.json").write_text(
            json.dumps(
                {
                    "datasets": {name: {} for name in ["mnist", "celeba-hq", "sgsm"]},
                    "file_count": 3,
                    "checksum_sidecar": "manifest-files.tsv",
                }
            )
        )

    def test_missing_corpus_fails_before_llm_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_missing_manifest(root)
            with self.assertRaisesRegex(DataContractError, "Incomplete RBT4DNN image corpus"):
                run_audit(root, samples_per_requirement=1)
            self.assertFalse((root / "experiments/llm-validity-audit/results.csv").exists())
            self.assertFalse((root / "experiments/llm-validity-audit/sample-manifest.csv").exists())

    def test_validate_rejects_incomplete_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_missing_manifest(root)
            with self.assertRaisesRegex(DataContractError, "missing 1"):
                validate_image_corpus(root, ["mnist"])

    def test_metadata_only_reproduction_succeeds_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            for name in ["requirements.csv", "manifest.json", "manifest-files.tsv"]:
                shutil.copy(ROOT / "data" / name, root / "data" / name)
            shutil.copytree(ROOT / "src", root / "src")
            shutil.copytree(ROOT / "scripts", root / "scripts")
            for name in [
                "replication",
                "precondition-validity",
                "mnist-lora-ablation",
                "cost-analysis",
            ]:
                source = ROOT / "experiments" / name
                destination = root / "experiments" / name
                destination.mkdir(parents=True)
                for python_file in source.glob("*.py"):
                    shutil.copy(python_file, destination / python_file.name)
            result = subprocess.run(
                [sys.executable, "scripts/reproduce.py"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "experiments/replication/results.csv").exists())
            self.assertEqual(
                (root / "experiments/precondition-validity/results.csv").read_bytes(),
                (ROOT / "experiments/precondition-validity/results.csv").read_bytes(),
            )
            self.assertEqual(
                (root / "experiments/mnist-lora-ablation/results.csv").read_bytes(),
                (ROOT / "experiments/mnist-lora-ablation/results.csv").read_bytes(),
            )

    def test_corpus_mode_fails_before_metadata_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_missing_manifest(root)
            shutil.copytree(ROOT / "src", root / "src")
            (root / "scripts").mkdir()
            (root / "experiments").mkdir()
            shutil.copy(ROOT / "scripts/reproduce.py", root / "scripts/reproduce.py")
            result = subprocess.run(
                [sys.executable, "scripts/reproduce.py", "--llm"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Incomplete RBT4DNN image corpus", result.stderr)
            self.assertFalse((root / "experiments/replication/results.csv").exists())


if __name__ == "__main__":
    unittest.main()
