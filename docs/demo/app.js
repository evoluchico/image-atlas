/* Image Atlas frontend.
 *
 * Render-only by design (DESIGN.md section 7): all filtering, aggregation
 * and representative selection happen in the backend. This file does view
 * math, fetches /api/viewport, and draws sprites from atlas pages.
 */
"use strict";

const FETCH_DEBOUNCE_MS = 80;
const GRID_TARGET_PX = 112;   // fixed on-screen cell size; sets the aggregation grid

// thumbnail display size — user-adjustable, and deliberately INDEPENDENT of the
// grid: making thumbnails bigger shows them bigger without re-aggregating
let markerPx = +(localStorage.getItem("atlasMarkerPx") || 42);

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");

let manifest = null;
let view = { cx: 0.5, cy: 0.5, upp: 0.001 }; // world units per CSS pixel
let uppFit = 0.001;
let filterToken = "all";         // token of the SQL filter
let selection = null;            // {token, count} from a lasso, or null
let lasso = null;                // screen-space points while drawing a lasso
let scene = { z: 0, aggregates: [], items: [] };
let markers = [];                // hit-test rects, topmost last
let hoverId = null, pinnedId = null;

function activeToken() {
  return selection ? selection.token : filterToken;
}
let fetchSeq = 0, fetchTimer = null;
let renderQueued = false;

/* ------------------------------------------------------------- view math */

function cssSize() {
  return { w: canvas.clientWidth, h: canvas.clientHeight };
}
function worldToScreen(x, y) {
  const { w, h } = cssSize();
  return [(x - view.cx) / view.upp + w / 2, (y - view.cy) / view.upp + h / 2];
}
function screenToWorld(sx, sy) {
  const { w, h } = cssSize();
  return [view.cx + (sx - w / 2) * view.upp, view.cy + (sy - h / 2) * view.upp];
}
function currentZ() {
  const z = Math.round(Math.log2(1 / (GRID_TARGET_PX * view.upp)));
  return Math.max(manifest.zoom.min, Math.min(manifest.zoom.max, z));
}

/* ----------------------------------------------------------- sprite cache
 * Sprites arrive as on-demand "strips": one WebP per viewport containing
 * exactly the sprites it needs, STRIP_COLS per row (must match server).
 */

const STRIP_COLS = 32;
const STRIP_MAX = 512;            // ids per request (server caps at 1024)
const spriteCache = new Map();    // id -> {img: Image|null, sx, sy}
let staticSheetLoading = false;

function ensureSprites(ids) {
  if (manifest.static) {
    // exported demo: every sprite lives in one prebuilt sheet
    if (staticSheetLoading) return;
    staticSheetLoading = true;
    fetch("./sheet.webp")
      .then((res) => res.blob())
      .then((blob) => createImageBitmap(blob))
      .then((img) => {
        const c = manifest.sprite_cell, sc = manifest.sheet_cols;
        for (let id = 0; id < manifest.count; id++) {
          spriteCache.set(id, { img, sx: (id % sc) * c, sy: Math.floor(id / sc) * c });
        }
        requestRender();
      });
    return;
  }
  const missing = ids.filter((id) => !spriteCache.has(id));
  if (!missing.length) return;
  if (spriteCache.size > 8000) spriteCache.clear();   // crude but sufficient
  const cell = manifest.sprite_cell;
  for (let i = 0; i < missing.length; i += STRIP_MAX) {
    const chunk = missing.slice(i, i + STRIP_MAX);
    const cols = Math.min(STRIP_COLS, chunk.length);
    chunk.forEach((id) => spriteCache.set(id, { img: null, sx: 0, sy: 0 }));
    fetch(`/api/sprites?ids=${chunk.join(",")}`)
      .then((res) => (res.ok ? res.blob() : Promise.reject()))
      .then((blob) => createImageBitmap(blob))
      .then((img) => {
        chunk.forEach((id, j) => {
          spriteCache.set(id, {
            img,
            sx: (j % cols) * cell,
            sy: Math.floor(j / cols) * cell,
          });
        });
        requestRender();
      })
      .catch(() => chunk.forEach((id) => spriteCache.delete(id)));
  }
}

/* -------------------------------------------------------------- fetching */

function scheduleFetch(immediate = false) {
  clearTimeout(fetchTimer);
  fetchTimer = setTimeout(doFetch, immediate ? 0 : FETCH_DEBOUNCE_MS);
}

async function doFetch() {
  const { w, h } = cssSize();
  const [x0, y0] = screenToWorld(-w * 0.05, -h * 0.05);
  const [x1, y1] = screenToWorld(w * 1.05, h * 1.05);
  const seq = ++fetchSeq;
  const params = new URLSearchParams({
    z: currentZ(),
    x0: Math.max(0, x0), y0: Math.max(0, y0),
    x1: Math.min(1, x1), y1: Math.min(1, y1),
    token: activeToken(),
  });
  try {
    const res = await fetch(`/api/viewport?${params}`);
    if (res.status === 410) {        // server restarted: token cache gone
      await applyFilter();
      return;
    }
    if (!res.ok) return;
    const data = await res.json();
    if (seq !== fetchSeq) return;    // stale response
    scene = data;
    ensureSprites([
      ...scene.aggregates.map((a) => a.id),
      ...scene.items.map((it) => it.id),
    ]);
    requestRender();
  } catch (e) { /* server gone; keep last scene */ }
}

/* ------------------------------------------------------------- rendering */

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}

function drawSprite(id, cxPx, cyPx, sizePx) {
  const entry = spriteCache.get(id);
  const cell = manifest.sprite_cell;
  const x = cxPx - sizePx / 2, y = cyPx - sizePx / 2;
  if (entry && entry.img) {
    ctx.drawImage(entry.img, entry.sx, entry.sy, cell, cell, x, y, sizePx, sizePx);
  } else {
    ctx.fillStyle = "#2b2f38";
    ctx.fillRect(x, y, sizePx, sizePx);
  }
  const active = id === pinnedId || id === hoverId;
  ctx.strokeStyle = active ? "#5b9dd9" : "rgba(0,0,0,.55)";
  ctx.lineWidth = active ? 2 : 1;
  ctx.strokeRect(x + 0.5, y + 0.5, sizePx - 1, sizePx - 1);
  return { x, y, w: sizePx, h: sizePx };
}

function drawBadge(text, xPx, yPx) {
  ctx.font = "11px system-ui, sans-serif";
  const tw = ctx.measureText(text).width;
  const bw = tw + 10, bh = 16;
  ctx.fillStyle = "rgba(20,22,27,.85)";
  ctx.beginPath();
  ctx.roundRect(xPx - bw / 2, yPx - bh / 2, bw, bh, 8);
  ctx.fill();
  ctx.strokeStyle = "#5b9dd9";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#e8ebf2";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, xPx, yPx + 0.5);
}

function fmtCount(n) {
  return n >= 10000 ? `${Math.round(n / 1000)}k`
       : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

function render() {
  const { w, h } = cssSize();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // world border
  const [bx0, by0] = worldToScreen(0, 0);
  const [bx1, by1] = worldToScreen(1, 1);
  ctx.strokeStyle = "#2a2e38";
  ctx.lineWidth = 1;
  ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);

  markers = [];
  const itemPx = Math.round(markerPx * 0.7);
  for (const it of scene.items) {
    const [sx, sy] = worldToScreen(it.x, it.y);
    const rect = drawSprite(it.id, sx, sy, itemPx);
    markers.push({ ...rect, id: it.id, count: 1 });
  }
  for (const ag of scene.aggregates) {
    const [sx, sy] = worldToScreen(ag.x, ag.y);
    const size = Math.min(markerPx * 1.55, markerPx + 7 * Math.log10(ag.count));
    const rect = drawSprite(ag.id, sx, sy, size);
    drawBadge(fmtCount(ag.count), rect.x + rect.w - 4, rect.y + 2);
    markers.push({ ...rect, id: ag.id, count: ag.count,
                   z: scene.z, tx: ag.tx, ty: ag.ty });
  }
  if (lasso && lasso.length > 1) {
    ctx.beginPath();
    ctx.moveTo(lasso[0][0], lasso[0][1]);
    for (const [px, py] of lasso) ctx.lineTo(px, py);
    ctx.closePath();
    ctx.fillStyle = "rgba(91,157,217,.12)";
    ctx.fill();
    ctx.strokeStyle = "#5b9dd9";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
  }
  hud.textContent =
    `z${scene.z} · ${scene.aggregates.length} aggregates · ${scene.items.length} items`;
  renderContents();   // repaint the side-panel grid as sprite strips arrive
}

/* ----------------------------------------------------------- interaction */

function hitTest(sx, sy) {
  for (let i = markers.length - 1; i >= 0; i--) {
    const m = markers[i];
    if (sx >= m.x && sx <= m.x + m.w && sy >= m.y && sy <= m.y + m.h) return m;
  }
  return null;
}

let dragging = false, dragMoved = false, lastMx = 0, lastMy = 0;

canvas.addEventListener("mousedown", (e) => {
  if (e.shiftKey) {                       // start a lasso selection
    const rect = canvas.getBoundingClientRect();
    lasso = [[e.clientX - rect.left, e.clientY - rect.top]];
    return;
  }
  dragging = true; dragMoved = false;
  lastMx = e.clientX; lastMy = e.clientY;
  canvas.classList.add("dragging");
});
window.addEventListener("mouseup", (e) => {
  if (lasso) {
    finishLasso();
    return;
  }
  if (!dragging) return;
  dragging = false;
  canvas.classList.remove("dragging");
  if (!dragMoved) {                       // click
    const rect = canvas.getBoundingClientRect();
    const m = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    pinnedId = m && m.id !== pinnedId ? m.id : null;
    if (m && pinnedId !== null && m.count > 1) openTileContents(m);
    else closeContents();
    updateDetail(pinnedId ?? hoverId);
    requestRender();
  }
});
canvas.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (lasso) {
    const [lx, ly] = lasso[lasso.length - 1];
    if (Math.hypot(mx - lx, my - ly) > 3) {
      lasso.push([mx, my]);
      requestRender();
    }
    return;
  }
  if (dragging) {
    const dx = e.clientX - lastMx, dy = e.clientY - lastMy;
    if (Math.abs(dx) + Math.abs(dy) > 2) dragMoved = true;
    view.cx -= dx * view.upp;
    view.cy -= dy * view.upp;
    lastMx = e.clientX; lastMy = e.clientY;
    requestRender();
    scheduleFetch();
    return;
  }
  const m = hitTest(mx, my);
  const id = m ? m.id : null;
  if (id !== hoverId) {
    hoverId = id;
    if (pinnedId === null) updateDetail(hoverId);
    requestRender();
  }
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const [wx, wy] = screenToWorld(mx, my);
  const factor = Math.exp(e.deltaY * 0.0012);
  view.upp = Math.min(uppFit * 2, Math.max(uppFit / 4096, view.upp * factor));
  const { w, h } = cssSize();
  view.cx = wx - (mx - w / 2) * view.upp;
  view.cy = wy - (my - h / 2) * view.upp;
  requestRender();
  scheduleFetch();
}, { passive: false });

window.addEventListener("resize", () => { requestRender(); scheduleFetch(); });

/* ------------------------------------------------------------ side panel */

const detailBox = document.getElementById("detail");
const detailImg = document.getElementById("detail-img");
const detailMeta = document.getElementById("detail-meta");
const detailPin = document.getElementById("detail-pin");
let detailSeq = 0;

async function updateDetail(id) {
  detailPin.textContent = pinnedId !== null ? "Pinned — click map background to unpin" : "";
  if (id === null || id === undefined) {
    detailBox.classList.add("hidden");
    return;
  }
  const seq = ++detailSeq;
  const res = await fetch(`/api/image/${id}`);
  if (!res.ok || seq !== detailSeq) return;
  const info = await res.json();
  detailBox.classList.remove("hidden");
  detailImg.src = info.preview_url;
  detailMeta.innerHTML = "";
  for (const [k, v] of Object.entries(info)) {
    if (k === "preview_url" || v === null) continue;
    const tr = document.createElement("tr");
    const td1 = document.createElement("td"), td2 = document.createElement("td");
    td1.textContent = k;
    td2.textContent = String(v);
    tr.append(td1, td2);
    detailMeta.append(tr);
  }
}

/* ------------------------------------------------------- tile contents grid
 * Clicking an aggregate opens a paginated grid of every image inside that
 * tile (current filter applied, best representatives first). Each page is
 * one /api/tile call + one sprite strip — bounded work regardless of size.
 */

const GRID_PAGE = 60;
const GRID_COLS = 6;
const contentsBox = document.getElementById("contents");
const contentsCount = document.getElementById("contents-count");
const contentsGrid = document.getElementById("contents-grid");
const contentsMore = document.getElementById("contents-more");
let contents = null;   // {z, tx, ty, total, ids, selected}

function closeContents() {
  contents = null;
  contentsBox.classList.add("hidden");
}

async function openTileContents(tile) {
  contents = { z: tile.z, tx: tile.tx, ty: tile.ty, total: 0, ids: [], selected: null };
  contentsBox.classList.remove("hidden");
  await loadMoreContents();
}

async function loadMoreContents() {
  if (!contents) return;
  const params = new URLSearchParams({
    z: contents.z, tx: contents.tx, ty: contents.ty,
    token: activeToken(), offset: contents.ids.length, limit: GRID_PAGE,
  });
  const res = await fetch(`/api/tile?${params}`);
  if (!res.ok || !contents) return;
  const d = await res.json();
  contents.total = d.total;
  contents.ids.push(...d.ids);
  ensureSprites(d.ids);
  contentsCount.textContent =
    `${contents.total.toLocaleString()} images in this region — showing ${contents.ids.length}`;
  contentsMore.classList.toggle("hidden", contents.ids.length >= contents.total);
  renderContents();
}

function renderContents() {
  if (!contents) return;
  const cellPx = 46, gap = 2;
  const rows = Math.ceil(contents.ids.length / GRID_COLS);
  const w = GRID_COLS * cellPx + (GRID_COLS - 1) * gap;
  const h = Math.max(1, rows * cellPx + (rows - 1) * gap);
  if (contentsGrid.width !== w || contentsGrid.height !== h) {
    contentsGrid.width = w;
    contentsGrid.height = h;
  }
  const g = contentsGrid.getContext("2d");
  g.clearRect(0, 0, w, h);
  const cell = manifest.sprite_cell;
  contents.ids.forEach((id, i) => {
    const x = (i % GRID_COLS) * (cellPx + gap);
    const y = Math.floor(i / GRID_COLS) * (cellPx + gap);
    const entry = spriteCache.get(id);
    if (entry && entry.img) {
      g.drawImage(entry.img, entry.sx, entry.sy, cell, cell, x, y, cellPx, cellPx);
    } else {
      g.fillStyle = "#2b2f38";
      g.fillRect(x, y, cellPx, cellPx);
    }
    if (id === contents.selected) {
      g.strokeStyle = "#5b9dd9";
      g.lineWidth = 2;
      g.strokeRect(x + 1, y + 1, cellPx - 2, cellPx - 2);
    }
  });
}

contentsGrid.addEventListener("click", (e) => {
  if (!contents) return;
  const rect = contentsGrid.getBoundingClientRect();
  const scale = contentsGrid.width / rect.width;
  const x = (e.clientX - rect.left) * scale;
  const y = (e.clientY - rect.top) * scale;
  const i = Math.floor(y / 48) * GRID_COLS + Math.floor(x / 48);
  const id = contents.ids[i];
  if (id === undefined) return;
  contents.selected = id;
  pinnedId = id;
  updateDetail(id);
  renderContents();
});

contentsMore.addEventListener("click", loadMoreContents);

/* ------------------------------------------------------------ size slider */

const sizeSlider = document.getElementById("size-slider");
const sizeVal = document.getElementById("size-val");
sizeSlider.value = markerPx;
sizeVal.textContent = `${markerPx}px`;
sizeSlider.addEventListener("input", () => {
  markerPx = +sizeSlider.value;
  sizeVal.textContent = `${markerPx}px`;
  localStorage.setItem("atlasMarkerPx", markerPx);
  requestRender();   // display size only; the grid is fixed, so no refetch
});

/* ------------------------------------------------------------- selection */

const selBox = document.getElementById("selection");
const selCount = document.getElementById("sel-count");

async function finishLasso() {
  const pts = lasso;
  lasso = null;
  requestRender();
  if (!pts || pts.length < 3) return;
  const polygon = pts.map(([sx, sy]) => screenToWorld(sx, sy));
  try {
    const res = await fetch("/api/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon, base_token: filterToken }),
    });
    if (!res.ok) return;
    selection = await res.json();
    closeContents();
    updateSelectionUI();
    scheduleFetch(true);
  } catch (e) { /* server gone */ }
}

function clearSelection() {
  selection = null;
  closeContents();
  updateSelectionUI();
  scheduleFetch(true);
}

function updateSelectionUI() {
  if (selection) {
    selBox.classList.remove("hidden");
    selCount.textContent = selection.count.toLocaleString();
  } else {
    selBox.classList.add("hidden");
  }
}

document.getElementById("sel-clear").addEventListener("click", clearSelection);
document.getElementById("sel-export").addEventListener("click", async () => {
  if (!selection) return;
  const res = await fetch(`/api/export?token=${selection.token}`);
  if (!res.ok) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(await res.blob());
  a.download = "atlas-export.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});

/* ---------------------------------------------------------------- filter */

const filterInput = document.getElementById("filter-input");
const filterStatus = document.getElementById("filter-status");
const filterError = document.getElementById("filter-error");

async function applyFilter() {
  filterError.textContent = "";
  const where = filterInput.value.trim();
  try {
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ where }),
    });
    const data = await res.json();
    if (!res.ok) {
      filterError.textContent = data.error || "filter failed";
      return;
    }
    filterToken = data.token;
    selection = null;            // a new filter supersedes any lasso
    closeContents();
    updateSelectionUI();
    filterStatus.textContent =
      `${data.count.toLocaleString()} / ${manifest.count.toLocaleString()} match`;
    scheduleFetch(true);
  } catch (e) {
    filterError.textContent = "server unreachable";
  }
}

document.getElementById("filter-apply").addEventListener("click", applyFilter);
document.getElementById("filter-clear").addEventListener("click", () => {
  filterInput.value = "";
  applyFilter();
});
filterInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); applyFilter(); }
});

/* ------------------------------------------------------------------ init */

async function init() {
  manifest = await (await fetch("/api/manifest")).json();
  document.getElementById("ds-name").textContent = manifest.name;
  document.getElementById("ds-count").textContent =
    `${manifest.count.toLocaleString()} images · layout: ${manifest.coords_source}`;
  document.title = `${manifest.name} — Image Atlas`;
  const { w, h } = cssSize();
  uppFit = 1.1 / Math.min(w, h);
  view = { cx: 0.5, cy: 0.5, upp: uppFit };
  requestRender();
  scheduleFetch(true);
}
init();
