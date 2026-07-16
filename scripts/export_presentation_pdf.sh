#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_html="${root_dir}/slides/index.html"
output_pdf="${1:-${root_dir}/slides/rbt4dnn-presentation.pdf}"

chrome_bin="${CHROME_BIN:-}"
if [[ -z "${chrome_bin}" && -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  chrome_bin="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi
if [[ -z "${chrome_bin}" ]]; then
  chrome_bin="$(find "${HOME}/Library/Caches/ms-playwright" -path '*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing' -type f -perm -111 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${chrome_bin}" || ! -x "${chrome_bin}" ]]; then
  echo "No Chromium executable found. Set CHROME_BIN to Chrome or Playwright Chromium." >&2
  exit 1
fi

mkdir -p "$(dirname "${output_pdf}")"
tmp_dir="$(mktemp -d)"
port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
python3 -m http.server "${port}" --bind 127.0.0.1 --directory "${root_dir}" >/dev/null 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT
sleep 0.5

"${chrome_bin}" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=5000 \
  --print-to-pdf="${tmp_dir}/full.pdf" \
  "http://127.0.0.1:${port}/slides/?full=1"

pdftoppm -jpeg -r 72 "${tmp_dir}/full.pdf" "${tmp_dir}/page" >/dev/null 2>&1
uv run python - "${tmp_dir}" "${output_pdf}" <<'PY'
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


temp_dir = Path(sys.argv[1])
output_pdf = Path(sys.argv[2])
pages = [Image.open(path).convert("RGB") for path in sorted(temp_dir.glob("page-*.jpg"))]
cream = "#e7e3da"
ink = "#68665f"
font = ImageFont.load_default(size=18)


def replace_region(page: Image.Image, box: tuple[int, int, int, int], label: str) -> None:
    draw = ImageDraw.Draw(page)
    draw.rectangle(box, fill=cream)
    draw.multiline_text(
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        f"{label}\nSource-linked in interactive deck",
        fill=ink,
        font=font,
        anchor="mm",
        align="center",
        spacing=7,
    )


for box, label in [
    ((58, 263, 373, 403), "MNIST artifact samples"),
    ((407, 263, 703, 403), "CelebA-HQ artifact samples"),
    ((738, 263, 1033, 403), "SGSM artifact samples"),
    ((1067, 263, 1363, 403), "ImageNet artifact samples"),
]:
    replace_region(pages[4], box, label)
replace_region(pages[5], (58, 217, 575, 610), "MNIST M3 artifact samples")

pages[0].save(
    output_pdf,
    "PDF",
    save_all=True,
    append_images=pages[1:],
    resolution=72,
    quality=90,
)
for page in pages:
    page.close()
PY

echo "Wrote publication-safe presentation PDF: ${output_pdf}"
