"""OCR build step: extract text from each image and index it for search.

Runs PaddleOCR over the bundle's own 512px previews (so it needs no access to
the original images), stores the recognized text in a SQLite FTS5 table keyed
by image id, and serves it via the unified search box. This is the expensive,
build-time half of OCR search — at serve time it is just an FTS5 MATCH.

Run from an env with paddleocr installed (kept separate from the core deps):
    python -m atlas ocr DATASET --lang pt [--no-gpu] [--limit N]
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def preview_path(dataset: Path, img_id: int) -> Path:
    return dataset / "previews" / f"{img_id // 1000:03d}" / f"{img_id:08d}.webp"


def _make_engine(kind: str, lang: str, gpu: bool):
    if kind == "easyocr":
        import easyocr
        langs = lang.split(",") if "," in lang else [lang]
        return ("easyocr", easyocr.Reader(langs, gpu=gpu))
    from paddleocr import PaddleOCR
    return ("paddle", PaddleOCR(lang=lang, use_angle_cls=False, show_log=False, use_gpu=gpu))


def _extract(engine, path: str) -> str:
    """Return concatenated recognized text for either backend."""
    kind, eng = engine
    if kind == "easyocr":
        return " ".join(eng.readtext(path, detail=0, paragraph=True))
    try:
        res = eng.ocr(path)
    except TypeError:
        res = eng.ocr(path, cls=False)
    out = []
    for page in res or []:
        for line in page or []:
            try:
                txt = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
            except (IndexError, TypeError):
                continue
            if txt:
                out.append(str(txt))
    return " ".join(out)


def run(args) -> None:
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{dataset} is not a dataset bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n = int(manifest["count"])

    engine = _make_engine(args.engine, args.lang, gpu=not args.no_gpu)
    print(f"OCR engine: {args.engine} (lang={args.lang}, gpu={not args.no_gpu})")

    con = sqlite3.connect(dataset / "metadata.sqlite")
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS ocr USING fts5(text)")
    done = {r[0] for r in con.execute("SELECT rowid FROM ocr")}   # resumable
    print(f"OCR '{manifest['name']}': {len(done)}/{n} already done")

    limit = args.limit or n
    processed = 0
    for img_id in range(n):
        if img_id in done:
            continue
        if processed >= limit:
            break
        p = preview_path(dataset, img_id)
        text = _extract(engine, str(p)) if p.exists() else ""
        con.execute("INSERT INTO ocr (rowid, text) VALUES (?, ?)", (img_id, text))
        processed += 1
        if processed % 200 == 0:
            con.commit()
            print(f"  ocr: {processed} this run ({img_id + 1}/{n})", end="\r", flush=True)
    con.commit()
    print(f"\n  ocr complete: {processed} processed this run")

    manifest["ocr"] = {"lang": args.lang}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    con.close()
    print("done")


def add_parser(sub) -> None:
    p = sub.add_parser("ocr", help="extract image text with PaddleOCR for search")
    p.add_argument("dataset", help="dataset bundle directory")
    p.add_argument("--engine", choices=["easyocr", "paddle"], default="easyocr")
    p.add_argument("--lang", default="en", help="language(s); easyocr accepts e.g. 'pt,en'")
    p.add_argument("--no-gpu", action="store_true", help="run OCR on CPU")
    p.add_argument("--limit", type=int, help="process at most N images this run (resumable)")
    p.set_defaults(func=run)
