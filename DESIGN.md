# Image Atlas — Design Document

A local-first tool for interactively exploring very large image collections
(target: up to 1,000,000 images) on a fixed 2D map, with metadata filtering
and representative-thumbnail aggregation.

This document is the authoritative spec for the implementation in this
folder. Section 1 records where the design came from (a ChatGPT-proposed
spec) and what was kept or changed, and why. Sections 2+ are the actual
design.

---

## 1. Relation to the original (ChatGPT) proposal

The starting point was a spec produced in a ChatGPT conversation
("Scaling CSN for 1M Images"). Its core ideas are sound and are kept:

**Kept (these are the right calls):**

1. **Hard build/serve split.** An offline *build* stage produces a
   self-contained dataset bundle; the *serve/GUI* stage only reads it and
   must fail loudly if the bundle is missing pieces. It never silently
   rebuilds anything.
2. **Fixed coordinates, no clustering.** Every image has a permanent
   (x, y) ∈ [0,1]². Filters never move points — they only change which
   points are visible. This preserves the user's spatial memory of the
   collection (the main UX advantage over hierarchical-cluster browsers).
3. **Quadtree tile aggregation as the only level-of-detail mechanism.**
   At zoom z, tile (tx, ty) = (⌊x·2^z⌋, ⌊y·2^z⌋). A tile with few
   filtered survivors shows them individually; a tile with many shows
   *one* representative thumbnail + a count badge. No cluster trees.
4. **Two image derivatives only:** a tiny map sprite packed into sprite
   atlases, and a mid-size detail preview for the side panel. Originals
   stay where they are.
5. **Summarization in the backend, never the frontend.** The frontend
   renders what it is given. The hot path per interaction is:
   filter-bitmap ∩ visible-tile members → count → pick representative.
   Nothing ever scans the full dataset at interaction time.
6. **Representative selection heuristic:** precomputed per-image
   `rep_score` (density + quality); at query time, weighted center of
   survivors → snap to a real survivor → maximize
   `rep_score − λ · distance_to_anchor`. Cheap, stable, good enough.

**Changed (with reasons):**

1. **No WebGL / PixiJS / React — plain Canvas2D + vanilla JS.**
   The original spec mandates "WebGL ONLY". That mandate contradicts the
   spec's own aggregation design: because every visible tile renders at
   most one aggregate marker or ≤ K items, the frontend never draws more
   than a few hundred sprites regardless of dataset size. Canvas2D
   `drawImage` from atlas pages handles that trivially at 60 fps, with a
   fraction of the code and zero build tooling. WebGL is the right answer
   only if you render all points as a point cloud — which this design
   deliberately avoids.
2. **No DuckDB — stdlib SQLite.** Metadata filtering is "arbitrary SQL
   WHERE over ~1M rows, returning IDs". SQLite does this comfortably
   (milliseconds with indexes, tens of ms worst-case full scan), ships
   with Python, and removes a native dependency. DuckDB would win on
   heavy analytical aggregation, which we don't do. The query interface
   (a SQL WHERE string) is identical, so swapping later is cheap.
3. **No Tauri/Electron packaging — a tiny stdlib HTTP server.**
   `atlas serve <dataset>` starts a local server (127.0.0.1) and the user
   opens a browser tab. The conversation itself conceded a browser tab is
   fine. Desktop packaging is an optional later layer, not architecture.
4. **No per-image sprite bookkeeping — sprite position is a pure
   function of the image ID.** IDs are dense (0..N−1), assigned in build
   order. Atlas page = `id // cells_per_page`, cell = `id % cells_per_page`.
   This removes `atlas_id / sprite_x / sprite_y` columns entirely; the
   frontend computes sprite UVs from the manifest constants.
5. **Tile "blobs" become per-zoom sorted index arrays.** Instead of one
   binary file per tile per zoom (up to 4^8 = 65k files at z=8, painful on
   every filesystem), the build writes, per zoom level: a permutation of
   image IDs sorted by tile key, plus the unique tile keys and their start
   offsets (CSR layout, three `.npy` files). The backend memory-maps them;
   slicing a tile is two array lookups. Same information, ~27 small files
   total instead of ~87k, and zero parsing code.
6. **Filter sets are dense boolean masks, not roaring bitmaps.** With
   dense IDs, a numpy `bool` array of length N (1 MB per million images)
   intersects with a tile slice via fancy indexing at memory speed. A
   compressed bitmap is an optimization we demonstrably don't need.
7. **Density via 2D histogram, not exact k-NN (k=15).** The density score
   only breaks ties when choosing a representative; exactness is
   irrelevant. A 512×512 histogram with light smoothing is O(N), needs no
   scipy/KD-tree, and produces visually equivalent choices.
8. **Detail previews are sharded** (`previews/000/00000123.webp`,
   1000 per directory). The original spec put up to 1M files in one
   directory, which degrades many filesystems and file browsers.
9. **Coordinates: bring-your-own, with a built-in fallback.** The build
   accepts precomputed 2D coordinates (the CSN workflow: external
   embeddings + UMAP). If absent, it computes a lightweight visual layout
   (PCA of downscaled color features) so the tool works out of the box —
   clearly labeled as a fallback, not a substitute for embedding
   projections. Computing CLIP embeddings is explicitly out of scope.
10. **Aggregate threshold default 8, configurable** (spec said 12).
    Items render at sprite size inside a ~112 px screen tile; beyond ~8
    they overlap into mush. Tunable in the manifest per dataset.

**Dropped from the original spec:** thumb 64/128/256 pyramids (earlier
drafts), bitmap caches per tag, cluster hints, embedding storage — all
either superseded by later decisions in the same conversation or YAGNI.

### Format v2 revision: sprite strips replace atlas pages

v1 packed sprites into shared 4096² WebP atlas pages fetched by the
frontend (sprite position a pure function of the image ID). Real-world
use at 406k images over an SSH tunnel exposed the flaw: sprites are
placed in build order, uncorrelated with map position, so **one
zoomed-out view touched 59 of 67 pages ≈ 370 MB of transfer** (pages of
busy photographs average ~6 MB), and the browser held ~4 GB of decoded
page bitmaps. No static placement fixes the zoomed-out case — visible
representatives always sample the whole map.

v2 stores raw sprites in `sprites.bin` (N × cell² × 3 bytes, O(1) row
access) and the server packs **exactly the sprites a viewport needs**
into one small WebP strip per request (`/api/sprites?ids=…`,
`STRIP_COLS = 32` per row, ≤ 1024 ids). A new view now transfers
~10–40 KB of JSON + one ~100–400 KB image, independent of dataset size;
the frontend caches sprites by ID and only requests missing ones. Strip
encode is ~1 ms at 1M scale; a startup daemon thread warms the OS page
cache over `sprites.bin` so cold-disk random reads don't dominate.
Cost: the bundle stores raw sprites (6.9 GB at 1M with 48 px cells) —
disk traded for interaction latency, the right trade for a local tool.

---

## 2. Goals and non-goals

**Goals**

- Smooth pan/zoom exploration of 10³–10⁶ images on a fixed 2D map.
- Metadata filtering via SQL WHERE; results update without moving points.
- One legible representative thumbnail + count per dense region.
- Hover/click → detail preview + metadata in a side panel.
- Fully local and offline. Dependencies: Python ≥ 3.10, numpy, Pillow.
  Nothing else. No build step for the frontend.

**Non-goals (v1)**

- Embedding computation, semantic search, captioning, OCR.
- Clustering of any kind.
- Web/cloud deployment, multi-user access.
- Editing/curation of the collection.

---

## 3. System overview

```
BUILD (offline, explicit)                    SERVE (runtime, read-only)
─────────────────────────                    ──────────────────────────
images dir / list                            dataset bundle
metadata CSV (optional)        ┌──────┐         │
coords CSV (optional)    ───►  │build │ ───►    ▼
                               └──────┘      ┌────────┐    HTTP     ┌─────────┐
                                dataset      │backend │ ◄────────►  │ browser │
                                bundle       │ SQLite │  /filter    │Canvas2D │
                                             │ numpy  │  /viewport  │ JS      │
                                             └────────┘  /image     └─────────┘
```

- `atlas build` — raw inputs → dataset bundle. May take minutes/hours.
- `atlas serve` — bundle → local HTTP server + browser UI. Starts in
  seconds, recomputes nothing, fails explicitly on a bad bundle.
- `atlas demo` — generates a synthetic test collection (for development
  and first-run experience).

---

## 4. Dataset bundle format (`format_version: 2`)

```
dataset/
  manifest.json               # all constants; see below
  metadata.sqlite             # table images(id PK, path, width, height, + user cols)
  sprites.bin                 # uint8 raw (N, cell, cell, 3) — O(1) row access
  points/
    xy.npy                    # float32 (N, 2), values in [0, 1]
    rep.npy                   # float32 (N,)  rep_score
    z{z}_order.npy            # uint32 (N,)   image ids sorted by tile key, per zoom
    z{z}_keys.npy             # uint32 (T,)   sorted unique tile keys (ty * 2^z + tx)
    z{z}_starts.npy           # uint32 (T+1,) CSR offsets into z{z}_order
    z{z}_rep.npy              # uint32 (T,)   precomputed unfiltered representative per tile
  previews/
    000/00000000.webp         # detail previews, max side 512, shard = id // 1000
    ...
```

`manifest.json`:

```json
{
  "format_version": 2,
  "name": "...",
  "count": 123456,
  "sprite_cell": 48,
  "zoom": {"min": 0, "max": 8},
  "aggregate_threshold": 8,
  "preview_max_side": 512,
  "coords_source": "provided | visual-pca | random",
  "metadata_columns": [{"name": "...", "type": "TEXT|REAL|INTEGER"}]
}
```

Invariants:

- IDs are dense `0..N−1`; row `id` in SQLite ↔ index `id` in every array
  ↔ row `id` in `sprites.bin`.
- Sprites are delivered to the frontend as per-viewport strips
  (section 6), never as whole files.
- `serve` validates: manifest present and version supported; all arrays
  present with length N; `sprites.bin` has exactly N·cell²·3 bytes;
  SQLite row count = N. Any mismatch → refuse to start, with a message
  naming the missing piece.

## 5. Build pipeline

Inputs: `--images DIR` (recursive; jpg/png/webp/bmp/tiff/gif),
optional `--metadata CSV` (joined on a filename/path column),
optional `--coords CSV` (path,x,y), `--out DIR`, plus tunables
(`--cell`, `--threshold`, `--max-zoom`, `--name`).

Steps (single process, vectorized where it matters):

1. **Scan & assign IDs.** Sort discovered paths for determinism; id =
   sort index.
2. **Decode once, derive twice.** Per image: load with Pillow → 48×48
   center-cropped sprite (written to its row in `sprites.bin`) →
   max-side-512 detail preview (WebP, quality 80) → 8×8 RGB feature
   vector (for fallback layout & quality score). Corrupt files are kept as gray
   placeholder sprites with `quality = 0` (IDs must stay dense; dropping
   files would desynchronize provided metadata/coords).
3. **Coordinates.** Provided CSV if given (validated, min-max normalized
   to [0.005, 0.995]); else PCA of the 8×8×3 features to 2D, rank-
   normalized per axis to spread the map (this is the documented
   fallback); else random (only if PCA degenerate).
4. **Scores.** `density` = bilinear sample of a lightly box-blurred
   512×512 histogram of point positions, rank-normalized to [0,1].
   `quality` = 1 if decoded and pixel variance above a floor, else
   0/0.25. `rep = 0.7 * density + 0.3 * quality`.
5. **Tile indexes.** For each z in 0..max_zoom: `key = ty * 2^z + tx`,
   stable argsort → `order`; `np.unique` on sorted keys → `keys`,
   `starts`. Additionally, the *unfiltered* representative of every tile
   is precomputed here (`z{z}_rep.npy`) with the same heuristic the
   server uses for filtered tiles (`atlas/summarize.py`) — the no-filter
   view is the most common one and is stable, so it should never be
   recomputed at interaction time. (~2.5 s total for z0..z8 at 1M.)
6. **Metadata DB.** `images(id INTEGER PRIMARY KEY, path TEXT, width
   INTEGER, height INTEGER, …user columns)`. User column types inferred
   (INTEGER/REAL/TEXT) from the CSV. Indexes on user columns are created
   only when a column has low cardinality (≤ 1% distinct) — cheap wins
   for tag-style filters.
7. **Manifest** written last; its presence marks a complete build.
   Builds are atomic-ish: output goes to `out/.building` then renamed.

Throughput expectation: image decoding dominates (~10–40 ms/image
single-threaded). Build parallelism (`--workers`) uses a process pool
for step 2 only; everything else is negligible.

## 6. Runtime: backend

A threaded stdlib `http.server`, bound to 127.0.0.1 by default
(`--host` can widen it; the server is unauthenticated, so remote viewing
is expected to go through SSH port forwarding — on headless machines the
browser launch is skipped and the `ssh -L` hint is printed instead).
State loaded at startup: manifest, memory-mapped numpy arrays, one
read-only SQLite connection per thread (`query_only` pragma), and a
small filter cache.

### Filtering

`POST /api/filter` `{"where": "<sql>"}` →
`{"token": "f3ab…", "count": 1234}`.

- Empty/absent WHERE → reserved token `all`.
- Execution: `SELECT id FROM images WHERE (<where>)` on a read-only,
  `query_only` connection → ids → dense `bool[N]` mask, cached under
  `token = sha1(where)[:12]` (LRU, 32 entries).
- SQL errors return 400 with SQLite's message (the UI shows it). This is
  a local single-user tool; the WHERE clause is the query language by
  design, and the connection cannot write.

### Viewport summarization

`GET /api/viewport?z=&x0=&y0=&x1=&y1=&token=` →

```json
{"z": 4,
 "aggregates": [{"tx":3,"ty":5,"count":412,"id":18211,"x":0.41,"y":0.66}],
 "items":      [{"id":7,"x":0.12,"y":0.30}]}
```

Algorithm, per request (z clamped to the manifest range; if the
requested span exceeds 4096 tiles the server lowers z until it fits and
reports the effective `z` in the response — the client's zoom-selection
rule keeps it to ~tens of tiles in practice):

```
mask = cache[token]                  # bool[N], or None for token "all"
for each tile key k in [ty0..ty1] × [tx0..tx1]:
    if mask is None:                            # unfiltered fast path
        n = starts[i+1] - starts[i]
        emit items if n <= threshold, else the PRECOMPUTED rep (z{z}_rep)
        continue
    members = order[starts[i]:starts[i+1]]      # via searchsorted(keys, k)
    surv    = members[mask[members]]
    n = len(surv)
    if n == 0: continue
    if n <= threshold: emit items (id, x, y)
    else:
        s  = stratified sample of surv, capped at 2048   # count n stays exact
        w  = rep[s]
        cx, cy = Σ(xy[s]·w) / Σw                # weighted center
        anchor = s[argmin ‖xy − (cx,cy)‖]       # snap to a real point
        best   = s[argmax (rep − λ·‖xy − xy[anchor]‖/tile_size)]   # λ = 0.25
        emit aggregate (tx, ty, n, best, xy[best])
```

The heuristic lives in `atlas/summarize.py`, shared verbatim by build
(precomputed unfiltered reps) and server (filtered tiles), so the two
cannot drift. The sample cap bounds per-tile math regardless of how
many survivors a filter leaves; reported counts are always exact.

### Lasso selection and export

`POST /api/select` `{"polygon": [[x,y],…], "base_token": "…"}` →
`{"token": "sel…", "count": 123}`.

Vectorized ray-casting point-in-polygon over a bounding-box prefilter,
intersected with the base token's mask. The result is cached as a token
in the same LRU as SQL filters — selections and filters are
interchangeable everywhere a token is accepted (viewport, export).

`GET /api/export?token=…` → CSV download of all metadata columns plus
`x, y` for every image in the token's set.

### Sprite strips

`GET /api/sprites?ids=3,17,99,…` (≤ 1024 ids) → one WebP image with the
requested sprites in row-major cells, `STRIP_COLS = 32` per row (a
constant shared with the frontend). The server gathers rows from the
memory-mapped `sprites.bin` and encodes once (~1 ms + WebP encode);
recent strips are LRU-cached, and responses carry `max-age` so the
browser caches repeat views. The frontend keeps an id→cell cache and
requests only sprites it hasn't seen, so panning back is free. See
"Format v2 revision" in section 1 for why this replaced atlas pages.

### Other endpoints

- `GET /api/manifest` — the manifest (frontend bootstrap).
- `GET /api/image/{id}` — metadata row + preview URL.
- `GET /previews/{shard}/{id}.webp` — static, long-cache headers
  (content is immutable per bundle).
- `GET /` + static frontend files.

## 7. Runtime: frontend

Single page, no framework, no build step: `index.html`, `app.js`,
`style.css`.

- **Map**: full-window canvas. View state = center (cx, cy) + scale
  (world-units-per-pixel). Drag to pan, wheel to zoom (anchored at
  cursor). World is [0,1]² with a subtle border.
- **Zoom→tile-level rule**: request
  `z = clamp(round(log2(canvasWidth / (112 · viewWidth))), zmin, zmax)`
  so a tile is ~112 px on screen — one marker per ~marker-sized cell.
- **Data flow**: on view change (debounced ~80 ms) fetch `/api/viewport`
  for the visible bounds (±half-tile margin) and current filter token;
  render the response. Stale responses (superseded by a newer request)
  are discarded.
- **Rendering**: sprites drawn from cached strip images via `drawImage`;
  after each viewport response the frontend batch-requests any sprites
  it hasn't cached (section 6) and re-renders as strips arrive.
  Aggregates: sprite + rounded count badge; slight size boost with
  log(count). Items: sprite at 60% size. Everything has a hit-test
  rectangle kept per frame.
- **Side panel**: filter input (SQL WHERE) + Apply + match count + error
  display; hover shows detail preview + metadata; click pins it (hover
  stops overriding until unpinned).
- **Lasso**: Shift-drag draws a freehand polygon; on release it is posted
  to `/api/select` with the current filter token as base. The returned
  selection token becomes the active viewport token; the panel shows the
  count with Export CSV / Clear buttons. Applying a new SQL filter
  clears the selection.
- **No logic beyond rendering**: the frontend never aggregates, filters,
  or scores anything — including the lasso, whose point-in-polygon test
  runs server-side.

## 8. CLI

```
python -m atlas build  --images DIR [--metadata CSV] [--coords CSV]
                       [--workers N] [--cell 48] [--max-zoom 8]
                       [--threshold 8] [--name NAME] --out DATASET
python -m atlas serve  DATASET [--port 8765] [--no-browser]
python -m atlas demo   --out DIR [--count 2000]   # synthetic dataset
```

`demo` writes a synthetic image collection (colored generative shapes in
several visual families) + metadata CSV + coherent coords, then runs
`build` on it — the smoke-test and first-run experience.

## 9. Performance budget (1M images)

Measured with `python tests/bench_scale.py` (fabricated 1M-point bundle,
clustered layout, this machine; "viewport worst case" = 256 tiles, the
full-world view at a 1280 px window):

| Piece | Size / measured time | Notes |
|---|---|---|
| xy + rep arrays | 12 MB | mmapped |
| tile indexes + reps (9 zooms) | ~40 MB | mmapped, CSR |
| dataset open + validate | 7 ms | |
| filter mask | 1 MB / filter | LRU-cached, warm hit ≈ 0 ms |
| SQLite indexed WHERE | 41 ms cold | once per filter, then cached |
| SQLite full-scan WHERE | 136 ms cold | worst case |
| viewport, no filter | 0.4 ms | precomputed tile reps |
| viewport, filtered | ≤ ~15 ms | mask gather dominates |
| lasso select (64 vertices) | ~14 ms | bbox prefilter + ray casting |
| sprite strip (250 sprites) | ~1 ms + WebP encode | one ~100–400 KB image per new view |
| sprites.bin | 6.9 GB on disk (48 px cells) | mmapped; warmed at startup |
| detail previews | ~30–60 KB each | fetched on hover only |

The invariant that guarantees this: **no code path at interaction time
touches more than (visible tiles ∩ their members) + one O(N) boolean
indexing op**, and per-tile representative math is bounded by the
2048-point sample cap.

## 10. Verification

- `python -m unittest discover tests` — unit tests (tile index vs brute
  force, point-in-polygon, type inference, sprite-position derivation
  checked against actual atlas pixels) and end-to-end tests on a built
  fixture (validation, filters, viewport count conservation across all
  zooms for filtered and unfiltered tokens, representative-is-a-survivor,
  selection ∩ filter, CSV export, broken-bundle rejection).
- `python tests/bench_scale.py` — fabricates a 1M-point bundle and
  measures the hot paths against the section 9 budget, including an
  end-to-end count-conservation assertion at 1M.

## 11. Future work (explicitly out of scope for now)

- Embedding/UMAP helper (`atlas embed`) as a separate optional command
  with its own heavy dependencies.
- Multiple projections per bundle (the CSN feature), switchable at
  runtime — format already leaves room: add `points/alt_<name>/`.
- Desktop packaging (pywebview/Tauri) if "browser tab" proves
  insufficient.
- Roaring bitmaps / DuckDB if profiling ever shows the mask or SQLite
  as a bottleneck (it won't at 1M).
