"""Export a dataset bundle as a fully static, backend-free demo page.

For SMALL datasets only (a few thousand images): all coordinates, scores
and metadata go into one data.json, all sprites into one sheet image, and
a fetch() shim (frontend/shim.js) answers the /api/* routes client-side
with the same algorithms the server uses. The real app.js runs unchanged,
so the demo behaves exactly like the installed tool.

Usage:
  python scripts/export_static_demo.py <dataset-bundle> <output-dir>
"""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas.server import Dataset  # noqa: E402

MAX_DEMO_IMAGES = 5000


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ds = Dataset(Path(sys.argv[1]))
    out = Path(sys.argv[2]).resolve()
    if ds.n > MAX_DEMO_IMAGES:
        sys.exit(f"{ds.n} images is too many for a static demo (max {MAX_DEMO_IMAGES})")
    out.mkdir(parents=True, exist_ok=True)

    # --- data.json: coordinates, scores, metadata ----------------------
    con = sqlite3.connect(f"file:{ds.root / 'metadata.sqlite'}?mode=ro", uri=True)
    cur = con.execute("SELECT * FROM images ORDER BY id")
    columns = [d[0] for d in cur.description if d[0] != "id"]
    rows = [list(r[1:]) for r in cur]
    con.close()

    sheet_cols = math.ceil(math.sqrt(ds.n))
    manifest = dict(ds.manifest)
    manifest["static"] = True
    manifest["sheet_cols"] = sheet_cols
    data = {
        "manifest": manifest,
        "x": [round(float(v), 5) for v in ds.xy[:, 0]],
        "y": [round(float(v), 5) for v in ds.xy[:, 1]],
        "rep": [round(float(v), 4) for v in ds.rep],
        "columns": columns,
        "rows": rows,
    }
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    # --- sprite sheet ---------------------------------------------------
    c = ds.cell
    rows_n = math.ceil(ds.n / sheet_cols)
    sheet = np.zeros((rows_n * c, sheet_cols * c, 3), np.uint8)
    for i in range(ds.n):
        r, col = divmod(i, sheet_cols)
        sheet[r * c : (r + 1) * c, col * c : (col + 1) * c] = ds.sprites[i]
    Image.fromarray(sheet).save(out / "sheet.webp", "WEBP", quality=80)

    # --- previews + frontend ---------------------------------------------
    if (out / "previews").exists():
        shutil.rmtree(out / "previews")
    shutil.copytree(ds.root / "previews", out / "previews")

    fe = ROOT / "atlas" / "frontend"
    shutil.copy(fe / "app.js", out / "app.js")
    shutil.copy(fe / "style.css", out / "style.css")
    shutil.copy(fe / "shim.js", out / "shim.js")
    html = (fe / "index.html").read_text(encoding="utf-8").replace(
        '<script src="./app.js"></script>',
        '<script src="./shim.js"></script>\n<script src="./app.js"></script>',
    )
    (out / "index.html").write_text(html, encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static demo: {ds.n} images -> {out}  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
