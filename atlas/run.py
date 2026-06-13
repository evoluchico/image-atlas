"""One-command quick start: build the dataset bundle if needed, then serve it.

    atlas run --images /path/to/photos

Builds into `<images>_atlas` next to the image folder on first use (slow,
once), then opens the viewer (instant on later runs).
"""
from __future__ import annotations

from pathlib import Path

from . import build, server


def run(args) -> None:
    images = Path(args.images).resolve()
    if not images.is_dir():
        raise SystemExit(f"--images: not a directory: {images}")
    out = Path(args.out).resolve() if args.out else images.parent / (images.name + "_atlas")

    if args.rebuild or not (out / "manifest.json").exists():
        args.out = str(out)
        args.force = args.rebuild
        build.run(args)
    else:
        print(f"Using existing dataset at {out}  (pass --rebuild to rebuild)")

    args.dataset = str(out)
    server.run(args)


def add_parser(sub) -> None:
    p = sub.add_parser("run", help="build if needed, then serve (the one-command path)")
    p.add_argument("--images", required=True, help="directory of images (recursive)")
    p.add_argument("--out", help="dataset directory (default: <images>_atlas next to it)")
    p.add_argument("--rebuild", action="store_true", help="rebuild even if the dataset exists")
    build.add_build_options(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1; use 0.0.0.0 for LAN "
                        "access — unauthenticated, prefer ssh -L port forwarding)")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.set_defaults(func=run)
