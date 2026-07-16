from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/casparbreloh/rbt4dnn-seminar.git"
PAGES_URL = "https://casparbreloh.github.io/rbt4dnn-seminar/"
EXPECTED_SLIDE_COUNT = 25
CORPUS_NOTEBOOKS = {
    "experiments/mnist-shared-generator/notebook.ipynb": {"mnist"},
    "experiments/celeba-shared-generator/notebook.ipynb": {"celeba-hq"},
    "experiments/llm-validity-audit/notebook.ipynb": {"mnist", "celeba-hq", "sgsm"},
}


class SlideParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.slide_count = 0
        self.local_targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "section" and "slide" in classes:
            self.slide_count += 1
        for name in ("href", "src"):
            target = attributes.get(name)
            if target and not urlparse(target).scheme and not target.startswith(("#", "//")):
                self.local_targets.append(target)


def check_manifest() -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    lines = (ROOT / "data" / manifest["checksum_sidecar"]).read_text().splitlines()
    assert len(lines) == manifest["file_count"] == 14_500

    paths: list[str] = []
    datasets: Counter[str] = Counter()
    for number, line in enumerate(lines, start=1):
        sha256, size, dataset, local_path, source_path = line.split("\t")
        assert len(sha256) == 64 and all(char in "0123456789abcdef" for char in sha256)
        assert int(size) > 0
        assert dataset in manifest["datasets"]
        for value, prefix in [
            (local_path, "data/images/"),
            (source_path, "data/images_from_loras/"),
        ]:
            path = PurePosixPath(value)
            assert value.startswith(prefix), f"unexpected manifest path on line {number}: {value}"
            assert not path.is_absolute() and ".." not in path.parts
        paths.append(local_path)
        datasets[dataset] += 1

    assert paths == sorted(paths), "manifest paths must stay sorted"
    assert len(paths) == len(set(paths)), "manifest paths must stay unique"
    assert datasets == Counter(
        {name: details["file_count"] for name, details in manifest["datasets"].items()}
    )


def check_readme_links() -> None:
    readme = (ROOT / "README.md").read_text()
    assert PAGES_URL in readme
    assert REPO_URL.removesuffix(".git") in readme
    for raw_target in re.findall(r"\[[^]]*]\(([^)]+)\)", readme):
        target = raw_target.split("#", 1)[0]
        parsed = urlparse(target)
        if parsed.scheme or target.startswith("#"):
            continue
        path = ROOT / unquote(target)
        assert path.exists(), f"README link does not exist: {raw_target}"


def check_notebooks() -> None:
    notebooks = sorted((ROOT / "experiments").glob("*/notebook.ipynb"))
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        assert REPO_URL in code, f"{path} does not clone the canonical repository"
        assert "data/requirements.csv" in code
        assert "origin/main" in code

        relative = str(path.relative_to(ROOT))
        required = CORPUS_NOTEBOOKS.get(relative)
        if required:
            assert "scripts/fetch_data.py" in code
            assert '"--verify"' in code
            for dataset in required:
                assert f'"{dataset}"' in code


def check_slides() -> None:
    site = ROOT / "slides"
    index = site / "index.html"
    html = index.read_text()
    parser = SlideParser()
    parser.feed(html)
    assert parser.slide_count == EXPECTED_SLIDE_COUNT
    assert '<meta name="viewport"' in html
    assert "keydown" in html and "touch" in html
    assert "contenteditable" in html

    targets = parser.local_targets
    targets += re.findall(r"url\(['\"]?([^)'\"]+)", html)
    for target in targets:
        if target.startswith("#"):
            continue
        parsed = urlparse(target)
        if parsed.scheme == "data":
            continue
        local = site / unquote(parsed.path)
        assert local.is_file(), f"slide asset does not exist: {target}"

    for asset in sorted((site / "assets").glob("*")):
        if asset.is_file() and asset.name != ".DS_Store":
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            assert digest in (site / "ASSET_PROVENANCE.md").read_text()


def check_public_pdf() -> None:
    pdf = ROOT / "slides/rbt4dnn-presentation.pdf"
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF-"), "public presentation PDF is missing or malformed"
    assert len(re.findall(rb"/Type\s*/Page\b", data)) == EXPECTED_SLIDE_COUNT
    assert len(data) > 100_000, "public presentation PDF is unexpectedly small"
    for marker in [b"/Users/", b"presentation-script", b"question-defense", b"study-guide"]:
        assert marker not in data, f"private marker embedded in public PDF: {marker!r}"


def check_tracked_privacy() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    private_prefixes = ("outputs/", "notizen/", ".context/", ".frontend-slides/")
    private_names = {"presentation-script.md", "question-defense.md", "study-guide.md"}
    for path in tracked:
        assert not path.startswith(private_prefixes), f"private path is tracked: {path}"
        assert Path(path).name not in private_names, f"private preparation file is tracked: {path}"


def main() -> None:
    check_manifest()
    check_readme_links()
    check_notebooks()
    check_slides()
    check_public_pdf()
    check_tracked_privacy()
    print(
        "Publication checks passed: manifest, links, notebooks, "
        f"and {EXPECTED_SLIDE_COUNT}-slide static site."
    )


if __name__ == "__main__":
    main()
