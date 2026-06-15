# Plan: frozen Windows .exe (Tier 2 — NOT yet implemented)

Goal: let a non-technical Windows user run Image Atlas **without installing
Python**. This is deferred; this file is the plan, not a built feature.

## Recommended approach: PyInstaller, core-only

Freeze the lightweight core (numpy + Pillow + stdlib) into a single
`atlas.exe`. Do **not** try to freeze the search/OCR extras — see below.

### Build sketch

- Add PyInstaller as a dev dependency.
- A spec file (`atlas.spec`) that:
  - entry point: `atlas/__main__.py`
  - `datas`: bundle `atlas/frontend/*` (so the server can serve the UI).
    PyInstaller puts them under `sys._MEIPASS`; `server.FRONTEND_DIR` must
    fall back to `Path(getattr(sys, "_MEIPASS", ...)) / "atlas/frontend"`
    when frozen. (One small code change, guard with `getattr`.)
  - `hiddenimports`: usually none beyond numpy; verify on a clean VM.
  - one-file (`--onefile`) for a single .exe, or one-dir for faster start.
- Build on a real Windows runner (GitHub Actions `windows-latest`) so the
  binary is native; cross-building from Linux is not supported.
- Smoke test in CI: `atlas.exe demo --out d & atlas.exe serve d --no-browser`.

### Distribution

- Attach `atlas.exe` to a GitHub Release (via `gh release create`).
- Optionally an installer (Inno Setup / NSIS) that adds a Start-menu entry
  and a file association, but a bare .exe is enough for v1.

## Why search/OCR are excluded from the freeze

- **Model weights download at runtime.** sentence-transformers and EasyOCR
  fetch model files (hundreds of MB) on first use. Freezing them means
  either (a) shipping a multi-GB exe, or (b) downloading on first run,
  which defeats "offline, no setup".
- **torch is huge and fragile to freeze** (CUDA libs, dynamic loads).
- Decision: the .exe covers **build + serve + filters + labels + the map**
  (the whole experience minus content search). Users who want CLIP/OCR use
  the `pip install "image-atlas[search]"` path, which is already the
  documented route and is unaffected by this.

## Open questions / risks

- Antivirus false positives on PyInstaller one-file exes are common; a
  signed binary avoids this but needs a code-signing cert (cost). Note in
  the release that SmartScreen may warn.
- First-run startup of one-file exes is slow (it unpacks to temp); one-dir
  is snappier if we ship a zip instead.
- Verify Pillow's bundled codecs (WebP) survive the freeze — WebP is core
  to the bundle format; add a CI assertion that `atlas.exe demo` produces
  and reads `.webp` sprites/previews.

## Effort estimate

~1 focused session: spec file + the `_MEIPASS` frontend-path guard +
Windows CI build/release job + a clean-VM smoke test. Low risk, but only
worth doing when there's a real non-technical Windows audience to hand it
to.
