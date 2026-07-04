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

// region labels + density underlay
let labels = [];                 // [{text, x, y, count, level}], importance-sorted
let maxDataLevel = 0;
let densityImg = null;
let contours = null;             // {levels: [{t, segments: [x0,y0,x1,y1,...]}]}
let showLabels = localStorage.getItem("atlasLabels") !== "0";
let showDensity = localStorage.getItem("atlasDensity") !== "0";

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const hud = document.getElementById("hud");

let manifest = null;
let view = { cx: 0.5, cy: 0.5, upp: 0.001 }; // world units per CSS pixel
let uppFit = 0.001;
let filterToken = "all";         // base token from the structured/SQL filters
let searchToken = null;          // token from a text search (layered on the filter)
let selection = null;            // {token, count} from a lasso, or null
let lasso = null;                // screen-space points while drawing a lasso
let scene = { z: 0, aggregates: [], items: [] };
let markers = [];                // hit-test rects, topmost last
let hoverId = null, pinnedId = null;
let anim = null;                 // zoom-level transition: stacks explode / implode
let animRAF = 0;

function activeToken() {
  return selection ? selection.token : (searchToken || filterToken);
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
  if (axisToken) params.set("axis", axisToken);
  try {
    const res = await fetch(`/api/viewport?${params}`);
    if (res.status === 410) {        // server restarted: token cache gone
      await applyFilter();
      return;
    }
    if (!res.ok) return;
    const data = await res.json();
    if (seq !== fetchSeq) return;    // stale response
    data._axis = axisToken || null;  // tag so a layout switch skips the animation
    const prev = scene;
    scene = data;
    setupTransition(prev, data);     // animate stacks splitting / merging on z change
    ensureSprites([
      ...scene.aggregates.map((a) => a.id),
      ...scene.items.map((it) => it.id),
    ]);
    requestRender();
  } catch (e) { /* server gone; keep last scene */ }
}

/* ------------------------------------------------------------- rendering */

function requestRender() {
  if (renderQueued || anim) return;   // while animating, the anim ticker drives render()
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}

function drawSprite(id, cxPx, cyPx, sizePx, alpha = 1) {
  const entry = spriteCache.get(id);
  const cell = manifest.sprite_cell;
  const x = cxPx - sizePx / 2, y = cyPx - sizePx / 2;
  if (alpha !== 1) ctx.globalAlpha = alpha;
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
  if (alpha !== 1) ctx.globalAlpha = 1;
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

// Density contour lines: drawn as vectors in screen space, so they stay crisp
// at any zoom (unlike the stretched raster). Higher levels = denser = brighter.
function drawContours() {
  const sx0 = worldToScreen(0, 0), sx1 = worldToScreen(1, 1);
  const w = sx1[0] - sx0[0], h = sx1[1] - sx0[1];
  const n = contours.levels.length;
  contours.levels.forEach((lvl, i) => {
    const seg = lvl.segments;
    if (!seg.length) return;
    ctx.beginPath();
    for (let k = 0; k < seg.length; k += 4) {
      ctx.moveTo(sx0[0] + seg[k] * w, sx0[1] + seg[k + 1] * h);
      ctx.lineTo(sx0[0] + seg[k + 2] * w, sx0[1] + seg[k + 3] * h);
    }
    const a = 0.22 + 0.5 * (i / Math.max(1, n - 1));
    ctx.strokeStyle = `rgba(232, 116, 184, ${a})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

// How many label levels to reveal at the current zoom: coarsest (level 0)
// always, one more level per zoom step in. Finer themes emerge as you zoom —
// the "breaks down into sub-regions" behaviour, from one flat label set.
function labelLevelCap() {
  const span = Math.log2(uppFit / view.upp);   // ~0 fully out, grows zooming in
  return Math.max(0, Math.min(maxDataLevel, Math.floor(span) - 1));
}

function renderLabels() {
  const { w, h } = cssSize();
  const cap = labelLevelCap();
  const placed = [];   // screen bboxes of already-drawn labels
  let drawn = 0;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const lab of labels) {            // labels are pre-sorted by importance
    if (lab.level > cap || drawn >= 40) continue;
    const [sx, sy] = worldToScreen(lab.x, lab.y);
    if (sx < -50 || sx > w + 50 || sy < -30 || sy > h + 30) continue;
    const fontPx = Math.max(11, Math.min(34, 13 + 6 * Math.log10(lab.count) - 3 * lab.level));
    ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
    const tw = ctx.measureText(lab.text).width;
    const box = { x: sx - tw / 2, y: sy - fontPx / 2, w: tw, h: fontPx };
    const pad = 6;
    if (placed.some((p) => box.x < p.x + p.w + pad && box.x + box.w + pad > p.x &&
                            box.y < p.y + p.h + pad && box.y + box.h + pad > p.y)) continue;
    placed.push(box);
    drawn++;
    const alpha = lab.level === 0 ? 1 : 0.82;
    ctx.lineWidth = Math.max(3, fontPx / 5);
    ctx.strokeStyle = `rgba(8, 9, 13, ${alpha})`;
    ctx.lineJoin = "round";
    ctx.strokeText(lab.text, sx, sy);    // dark halo for legibility over thumbnails
    ctx.fillStyle = `rgba(238, 241, 248, ${alpha})`;
    ctx.fillText(lab.text, sx, sy);
  }
}

/* ------------------------------------------------- stack explode / implode
 * When a zoom step changes the aggregation level, the whole scene is swapped
 * at once. Instead of a hard cut we animate: on zoom-IN each new (finer) stack
 * flies out from the coarse stack it emerged from; on zoom-OUT the old finer
 * stacks fly inward and fade into the coarse stack that absorbs them. The
 * parent/child link is derived purely from world coordinates — a marker at
 * (x, y) sits in tile floor(x·2^z), so its ancestor at a coarser z is just
 * floor(x·2^coarseZ). No backend fields needed.
 */
const ANIM_MS = 300;
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const easeOut = (t) => 1 - Math.pow(1 - t, 3);

function markerSizeAgg(count) {
  return Math.min(markerPx * 1.55, markerPx + 7 * Math.log10(count));
}

function tileKey(x, y, z) {
  const n = Math.pow(2, z);
  return `${Math.floor(x * n)},${Math.floor(y * n)}`;
}

// Flatten a scene into drawable markers, each carrying its on-screen size, so a
// child can grow out of / shrink into its parent's actual footprint.
function sceneMarkers(s) {
  const out = [];
  const itemPx = Math.round(markerPx * 0.7);
  for (const it of s.items) out.push({ id: it.id, x: it.x, y: it.y, size: itemPx });
  for (const ag of s.aggregates)
    out.push({ id: ag.id, x: ag.x, y: ag.y, size: markerSizeAgg(ag.count) });
  return out;
}

function setupTransition(prev, next) {
  if (animRAF) { cancelAnimationFrame(animRAF); animRAF = 0; }
  anim = null;
  if (!prev || prev._axis !== next._axis) return;         // layout switch: no anim
  if (prev.z === next.z) return;                          // pan / same level: no anim
  const prevEmpty = !prev.aggregates.length && !prev.items.length;
  const nextEmpty = !next.aggregates.length && !next.items.length;
  if (prevEmpty || nextEmpty) return;                     // first paint / cleared: no anim
  const dir = next.z > prev.z ? "in" : "out";
  const coarseZ = Math.min(prev.z, next.z);
  const coarse = new Map();                               // ancestor tile -> coarse marker
  for (const m of sceneMarkers(dir === "in" ? prev : next))
    coarse.set(tileKey(m.x, m.y, coarseZ), m);
  anim = { start: performance.now(), dir, coarseZ, coarse,
           ghosts: dir === "out" ? sceneMarkers(prev) : null };
  startAnimTicker();
}

function startAnimTicker() {
  if (animRAF) return;
  const tick = () => {
    animRAF = 0;
    if (!anim) return;
    const done = (performance.now() - anim.start) >= ANIM_MS;
    if (done) anim = null;                                // last frame lands at rest
    render();
    if (!done) animRAF = requestAnimationFrame(tick);
  };
  animRAF = requestAnimationFrame(tick);
}

// Screen-space placement for one current-scene marker, factoring the animation.
function animMarker(x, y, base) {
  const [sx, sy] = worldToScreen(x, y);
  if (!anim) return { sx, sy, size: base, alpha: 1 };
  const e = easeOut(clamp((performance.now() - anim.start) / ANIM_MS, 0, 1));
  if (anim.dir === "in") {
    const p = anim.coarse.get(tileKey(x, y, anim.coarseZ));
    if (!p) return { sx, sy, size: base, alpha: e };      // no ancestor: just fade in
    const [px, py] = worldToScreen(p.x, p.y);
    return { sx: lerp(px, sx, e), sy: lerp(py, sy, e),
             size: lerp(p.size, base, e), alpha: 1 };      // fly out & grow from parent
  }
  return { sx, sy, size: base, alpha: e };                // zoom-out: coarse fades in place
}

// Zoom-out only: draw the outgoing finer stacks sliding into their absorber.
function drawGhosts() {
  const e = easeOut(clamp((performance.now() - anim.start) / ANIM_MS, 0, 1));
  for (const g of anim.ghosts) {
    const [gx, gy] = worldToScreen(g.x, g.y);
    const dest = anim.coarse.get(tileKey(g.x, g.y, anim.coarseZ));
    let sx = gx, sy = gy, size = g.size;
    if (dest) {
      const [dx, dy] = worldToScreen(dest.x, dest.y);
      sx = lerp(gx, dx, e); sy = lerp(gy, dy, e);
      size = lerp(g.size, dest.size, e);
    }
    drawSprite(g.id, sx, sy, size, 1 - e);
  }
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

  const scatter = axisMode === "scatter";   // axis plane replaces the UMAP layout
  // density underlay (backmost): soft raster glow + crisp vector contour lines
  if (showDensity && densityImg && !scatter) {
    ctx.globalAlpha = 0.6;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(densityImg, bx0, by0, bx1 - bx0, by1 - by0);
    ctx.globalAlpha = 1;
  }
  if (showDensity && contours && !scatter) drawContours();

  ctx.strokeStyle = "#2a2e38";
  ctx.lineWidth = 1;
  ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);

  markers = [];
  if (anim && anim.dir === "out") drawGhosts();   // outgoing stacks sliding inward
  const overlay = axisMode === "overlay";
  const itemPx = Math.round(markerPx * 0.7);
  for (const it of scene.items) {
    const a = animMarker(it.x, it.y, itemPx);
    const rect = drawSprite(it.id, a.sx, a.sy, a.size, a.alpha);
    if (overlay && it.score != null) outlineByScore(rect, it.score, a.alpha);
    drawPickRing(rect, it.id);
    const [fx, fy] = worldToScreen(it.x, it.y);   // hit-test at rest position
    markers.push({ x: fx - itemPx / 2, y: fy - itemPx / 2, w: itemPx, h: itemPx,
                   id: it.id, count: 1 });
  }
  for (const ag of scene.aggregates) {
    const size = markerSizeAgg(ag.count);
    const a = animMarker(ag.x, ag.y, size);
    const rect = drawSprite(ag.id, a.sx, a.sy, a.size, a.alpha);
    if (overlay && ag.score != null) outlineByScore(rect, ag.score, a.alpha);
    drawPickRing(rect, ag.id);
    if (a.alpha < 1) ctx.globalAlpha = a.alpha;
    drawBadge(fmtCount(ag.count), rect.x + rect.w - 4, rect.y + 2);
    if (a.alpha < 1) ctx.globalAlpha = 1;
    const [fx, fy] = worldToScreen(ag.x, ag.y);
    markers.push({ x: fx - size / 2, y: fy - size / 2, w: size, h: size,
                   id: ag.id, count: ag.count, z: scene.z, tx: ag.tx, ty: ag.ty });
  }
  if (scatter) drawQuadrants();
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
  if (showLabels && labels.length && axisMode !== "scatter") renderLabels();
  hud.textContent =
    `z${scene.z} · ${scene.aggregates.length} aggregates · ${scene.items.length} items`;
  renderContents();   // repaint the side-panel grid as sprite strips arrive
  if (axisMode === "overlay" && axisToken) renderAxisStrip();   // strip fills in too
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
    if (axisPickSlot) {                    // picking images for an axis end
      if (m) togglePick(axisPickSlot, m.id);
      return;
    }
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
  contents = { kind: "tile", z: tile.z, tx: tile.tx, ty: tile.ty,
               total: 0, ids: [], selected: null };
  contentsBox.classList.remove("hidden");
  await loadMoreContents();
}

// ranked search results, shown in the same grid as stack contents
function showSearchResults(ids, count, exact) {
  contents = { kind: "search", total: count, ids: ids.slice(), selected: null };
  contentsBox.classList.remove("hidden");
  contentsCount.textContent = exact
    ? `${count.toLocaleString()} exact matches — showing top ${ids.length}`
    : `top ${ids.length} most relevant`;
  contentsMore.classList.add("hidden");
  ensureSprites(ids);
  renderContents();
}

async function loadMoreContents() {
  if (!contents || contents.kind !== "tile") return;
  const params = new URLSearchParams({
    z: contents.z, tx: contents.tx, ty: contents.ty,
    token: activeToken(), offset: contents.ids.length, limit: GRID_PAGE,
  });
  if (axisToken) params.set("axis", axisToken);
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
      body: JSON.stringify({ polygon, base_token: searchToken || filterToken }),
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
const filterControls = document.getElementById("filter-controls");

async function postFilter(body) {
  filterError.textContent = "";
  try {
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      filterError.textContent = data.error || "filter failed";
      return;
    }
    filterToken = data.token;
    selection = null;            // a new filter supersedes any lasso
    if (searchToken) reRunSearch();   // re-apply the search on the new filter base
    closeContents();
    updateSelectionUI();
    filterStatus.textContent =
      `${data.count.toLocaleString()} / ${manifest.count.toLocaleString()} match`;
    scheduleFetch(true);
  } catch (e) {
    filterError.textContent = "server unreachable";
  }
}

const applyFilter = () => postFilter({ where: filterInput.value.trim() });

// Build a structured-filter spec from the rendered controls (no SQL typed).
function applyStructured() {
  filterInput.value = "";   // structured controls and the SQL box are exclusive
  const filters = [];
  for (const fc of filterControls.querySelectorAll(".fc")) {
    const col = fc.dataset.col;
    const sel = fc.querySelector("select");
    if (sel) {
      const values = [...sel.selectedOptions].map((o) => o.value).filter((v) => v !== "");
      if (values.length) filters.push({ col, values });
    } else {
      const lo = fc.querySelector(".fc-min").value;
      const hi = fc.querySelector(".fc-max").value;
      const f = { col };
      if (lo !== "") f.min = Number(lo);
      if (hi !== "") f.max = Number(hi);
      if ("min" in f || "max" in f) filters.push(f);
    }
  }
  postFilter({ filters });
}

function renderFilterControls(columns) {
  filterControls.innerHTML = "";
  for (const c of columns) {
    if (c.kind === "text") continue;   // free text: needs search, not a control
    const fc = document.createElement("div");
    fc.className = "fc";
    fc.dataset.col = c.name;
    const name = document.createElement("div");
    name.className = "fc-name";
    name.textContent = c.name;
    fc.appendChild(name);
    if (c.kind === "choice") {
      const sel = document.createElement("select");
      sel.multiple = c.values.length > 1;
      sel.size = Math.min(6, Math.max(2, c.values.length));
      for (const v of c.values) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        sel.appendChild(o);
      }
      sel.addEventListener("change", applyStructured);
      fc.appendChild(sel);
    } else {                            // range
      const row = document.createElement("div");
      row.className = "fc-range";
      row.innerHTML =
        `<input type="number" class="fc-min" placeholder="${c.min}">` +
        `<span class="muted">–</span>` +
        `<input type="number" class="fc-max" placeholder="${c.max}">`;
      for (const inp of row.querySelectorAll("input")) {
        inp.addEventListener("change", applyStructured);
      }
      fc.appendChild(row);
    }
    filterControls.appendChild(fc);
  }
  if (!filterControls.children.length) {
    filterControls.innerHTML =
      '<span class="muted">No filterable metadata columns.</span>';
  }
}

async function loadColumns() {
  try {
    const { columns } = await (await fetch("/api/columns")).json();
    renderFilterControls(columns);
  } catch (e) { /* leave empty */ }
}

/* ---------------------------------------------------------------- search */

const searchInput = document.getElementById("search-input");
const searchStatus = document.getElementById("search-status");
const searchExact = document.getElementById("search-exact");

async function runSearch() {
  const q = searchInput.value.trim();
  if (!q) { clearSearch(); return; }
  const mode = searchExact.checked ? "text" : "fused";
  searchStatus.textContent = "searching…";
  try {
    const res = await fetch(
      `/api/search?q=${encodeURIComponent(q)}&base=${filterToken}&mode=${mode}`);
    const data = await res.json();
    if (!res.ok) { searchStatus.textContent = data.error || "search failed"; return; }
    searchToken = data.token;
    selection = null;
    updateSelectionUI();
    searchStatus.textContent = data.exact
      ? `${data.count.toLocaleString()} matches`
      : `top ${data.count.toLocaleString()} by relevance`;
    showSearchResults(data.ids || [], data.count, data.exact);  // ranked grid
    scheduleFetch(true);
  } catch (e) { searchStatus.textContent = "server unreachable"; }
}

const reRunSearch = runSearch;   // re-run when the filter base changes

function clearSearch() {
  searchToken = null;
  searchInput.value = "";
  searchStatus.textContent = "";
  if (contents && contents.kind === "search") closeContents();
  scheduleFetch(true);
}

document.getElementById("search-go").addEventListener("click", runSearch);
document.getElementById("search-clear").addEventListener("click", clearSearch);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); runSearch(); }
});
searchExact.addEventListener("change", () => {
  if (searchInput.value.trim()) runSearch();
});

document.getElementById("filter-apply").addEventListener("click", applyFilter);
document.getElementById("filter-clear").addEventListener("click", () => {
  filterInput.value = "";
  searchToken = null; searchInput.value = ""; searchStatus.textContent = "";
  for (const sel of filterControls.querySelectorAll("select")) sel.selectedIndex = -1;
  for (const inp of filterControls.querySelectorAll("input")) inp.value = "";
  postFilter({ where: "" });
});
filterInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); applyFilter(); }
});

/* ------------------------------------------------------------ semantic axes
 * Define an A↔B direction in CLIP embedding space (by text, hand-picked images,
 * or lasso groups). One axis tints the map + fills a spectrum strip; two axes
 * relayout the map into a quadrant plot. All scoring is a matvec server-side.
 */
let axisToken = null;
let axisMode = null;        // "overlay" | "scatter"
let axisPayload = null;     // server response: {labels, divX, divY, stats, spectrum}
let axisPickSlot = null;    // "xa"|"xb"|"ya"|"yb" while picking images on the map
const mkBasket = () => ({ ids: new Set(), tokens: [] });
const axisBaskets = { xa: mkBasket(), xb: mkBasket(), ya: mkBasket(), yb: mkBasket() };
const AXIS_SLOTS = { x: ["xa", "xb"], y: ["ya", "yb"] };
const SLOT_DEFAULT = { xa: "A", xb: "B", ya: "C", yb: "D" };

// diverging colour ramp: 0 = end B (blue) · 0.5 = neutral · 1 = end A (red)
function divergingColor(t) {
  t = Math.max(0, Math.min(1, t));
  const A = [59, 111, 214], M = [130, 134, 150], B = [214, 75, 59];
  const mix = (u, v, k) => Math.round(u + (v - u) * k);
  const c = t < 0.5 ? A.map((v, i) => mix(v, M[i], t * 2))
                    : M.map((v, i) => mix(v, B[i], (t - 0.5) * 2));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// Overlay mode: a coloured ring just inside the thumbnail's edge (no fill), so
// the image stays fully visible and the axis reads as an outline colour.
function outlineByScore(rect, score, alpha) {
  ctx.globalAlpha = alpha ?? 1;
  ctx.strokeStyle = divergingColor(score);
  ctx.lineWidth = 2.5;
  ctx.strokeRect(rect.x + 1.5, rect.y + 1.5, rect.w - 3, rect.h - 3);
  ctx.globalAlpha = 1;
}
function drawPickRing(rect, id) {
  if (!axisPickSlot || !axisBaskets[axisPickSlot].ids.has(id)) return;
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 3;
  ctx.strokeRect(rect.x - 1, rect.y - 1, rect.w + 2, rect.h + 2);
}

// Quadrant crosshair + the four end-labels at the plane edges (scatter mode).
function drawQuadrants() {
  const { w, h } = cssSize();
  const dx = axisPayload && axisPayload.divX != null ? axisPayload.divX : 0.5;
  const dy = axisPayload && axisPayload.divY != null ? axisPayload.divY : 0.5;
  const sx = worldToScreen(dx, 0)[0];
  const sy = worldToScreen(0, dy)[1];
  ctx.strokeStyle = "rgba(232,236,244,.35)";
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(w, sy); ctx.stroke();
  ctx.setLineDash([]);
  const L = axisPayload && axisPayload.labels;
  if (!L) return;
  ctx.font = "600 13px system-ui, sans-serif";
  const edge = (text, x, y, align, baseline) => {
    if (!text) return;
    ctx.textAlign = align; ctx.textBaseline = baseline;
    ctx.lineWidth = 3; ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(8,9,13,.85)";
    ctx.strokeText(text, x, y);
    ctx.fillStyle = "#eef1f8";
    ctx.fillText(text, x, y);
  };
  edge(L.x && L.x.a, w - 10, h / 2, "right", "middle");   // high X → right
  edge(L.x && L.x.b, 10, h / 2, "left", "middle");
  edge(L.y && L.y.a, w / 2, 8, "center", "top");          // high Y → top
  edge(L.y && L.y.b, w / 2, h - 8, "center", "bottom");
}

const endEl = (slot) => document.querySelector(`.axis-end[data-slot="${slot}"]`);
const endText = (slot) => endEl(slot).querySelector(".axis-text").value.trim();
const slotName = (slot) => endText(slot) || SLOT_DEFAULT[slot];
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function setAxisStatus(msg, err) {
  const el = document.getElementById("axis-status");
  el.textContent = msg || "";
  el.classList.toggle("err", !!err);
}

function buildEndRows() {
  for (const [axis, slots] of Object.entries(AXIS_SLOTS)) {
    const container = document.getElementById(`axis-${axis}`);
    for (const slot of slots) {
      const div = document.createElement("div");
      div.className = "axis-end";
      div.dataset.slot = slot;
      div.innerHTML =
        `<input class="axis-text" type="text" ` +
          `placeholder="End ${SLOT_DEFAULT[slot]}: type a description, or pick images →">` +
        `<div class="axis-tools">` +
          `<button class="axis-pick secondary" type="button">pick</button>` +
          `<button class="axis-addsel secondary" type="button">+ selection</button>` +
          `<button class="axis-clearend secondary" type="button">clear</button>` +
          `<span class="axis-count muted"></span></div>`;
      container.appendChild(div);
      div.querySelector(".axis-pick").addEventListener("click", () => togglePickSlot(slot));
      div.querySelector(".axis-addsel").addEventListener("click", () => addSelectionToEnd(slot));
      div.querySelector(".axis-clearend").addEventListener("click", () => clearEnd(slot));
      div.querySelector(".axis-text").addEventListener("input", () => updateEndCount(slot));
    }
  }
}

function togglePickSlot(slot) {
  axisPickSlot = axisPickSlot === slot ? null : slot;
  for (const b of document.querySelectorAll(".axis-pick")) b.classList.remove("picking");
  if (axisPickSlot) {
    endEl(axisPickSlot).querySelector(".axis-pick").classList.add("picking");
    setAxisStatus(`Picking for End ${SLOT_DEFAULT[slot]} — click images (pan freely across the map); click again to remove.`);
  } else {
    setAxisStatus("");
  }
  requestRender();
}
function togglePick(slot, id) {
  const b = axisBaskets[slot];
  if (b.ids.has(id)) b.ids.delete(id); else b.ids.add(id);
  updateEndCount(slot);
  requestRender();
}
function addSelectionToEnd(slot) {
  if (!selection) { setAxisStatus("Lasso a group first (Shift-drag), then add it.", true); return; }
  const b = axisBaskets[slot];
  if (!b.tokens.includes(selection.token)) b.tokens.push(selection.token);
  updateEndCount(slot);
  setAxisStatus(`Added ${selection.count.toLocaleString()} images to End ${SLOT_DEFAULT[slot]}.`);
}
function clearEnd(slot) {
  axisBaskets[slot] = mkBasket();
  endEl(slot).querySelector(".axis-text").value = "";
  updateEndCount(slot);
  requestRender();
}
function updateEndCount(slot) {
  const el = endEl(slot).querySelector(".axis-count");
  if (endText(slot)) { el.textContent = "text"; return; }
  const b = axisBaskets[slot];
  const parts = [];
  if (b.ids.size) parts.push(`${b.ids.size} picked`);
  if (b.tokens.length) parts.push(`${b.tokens.length} group${b.tokens.length > 1 ? "s" : ""}`);
  el.textContent = parts.join(" + ");
}
function endSpec(slot) {
  const txt = endText(slot);
  if (txt) return { text: txt, label: txt };
  const b = axisBaskets[slot];
  if (b.ids.size || b.tokens.length)
    return { ids: [...b.ids], tokens: b.tokens.slice(), label: slotName(slot) };
  return null;
}

function resetView() { view = { cx: 0.5, cy: 0.5, upp: uppFit }; }

async function buildAxis() {
  const xa = endSpec("xa");
  if (!xa) { setAxisStatus("Axis 1 needs at least End A.", true); return; }
  const body = { x: { a: xa, b: endSpec("xb") }, base_token: filterToken };
  const ya = endSpec("ya");
  if (ya) body.y = { a: ya, b: endSpec("yb") };
  axisPickSlot = null;
  for (const b of document.querySelectorAll(".axis-pick")) b.classList.remove("picking");
  setAxisStatus("Building…");
  try {
    const res = await fetch("/api/axis", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) { setAxisStatus(data.error || "axis failed", true); return; }
    axisToken = data.token;
    axisMode = data.mode;
    axisPayload = data;
    if (axisMode === "scatter") resetView();
    updateAxisChrome();
    scheduleFetch(true);
  } catch (e) { setAxisStatus("server unreachable", true); }
}
function clearAxis() {
  const wasScatter = axisMode === "scatter";
  axisToken = null; axisMode = null; axisPayload = null;
  updateAxisChrome();
  if (wasScatter) resetView();
  scheduleFetch(true);
}

function updateAxisChrome() {
  const strip = document.getElementById("axis-strip");
  const legend = document.getElementById("axis-legend");
  document.getElementById("axis-export").classList.toggle("hidden", !axisToken);
  if (!axisToken) {
    strip.classList.add("hidden");
    legend.classList.add("hidden");
    setAxisStatus("");
    return;
  }
  const L = axisPayload.labels;
  if (axisMode === "overlay") {
    legend.classList.remove("hidden");
    legend.innerHTML =
      `<span>${escapeHtml(L.x.b)}</span><span class="bar"></span><span>${escapeHtml(L.x.a)}</span>`;
    document.getElementById("axis-strip-a").textContent = L.x.a;
    document.getElementById("axis-strip-b").textContent = L.x.b;
    strip.classList.remove("hidden");
    renderAxisStrip();
    setAxisStatus("Overlay: thumbnails outlined by the axis; strip shows the A↔B range.");
  } else {
    legend.classList.add("hidden");
    strip.classList.add("hidden");
    setAxisStatus("Quadrant plot: X and Y are your two axes. Drag/zoom as usual.");
  }
}

// Spectrum strip: the k representative thumbnails placed left→right by score.
function renderAxisStrip() {
  const spec = (axisPayload && axisPayload.spectrum) || [];
  if (!spec.length) return;
  ensureSprites(spec.map((s) => s.id));
  const cv = document.getElementById("axis-strip-canvas");
  const rectW = cv.clientWidth || 300, rectH = cv.clientHeight || 52;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(rectW * dpr)) cv.width = Math.round(rectW * dpr);
  if (cv.height !== Math.round(rectH * dpr)) cv.height = Math.round(rectH * dpr);
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, rectW, rectH);
  const cell = manifest.sprite_cell;
  const sz = rectH - 4;
  for (const s of spec) {
    const x = 2 + s.pos * (rectW - sz - 4), y = 2;
    const e = spriteCache.get(s.id);
    if (e && e.img) g.drawImage(e.img, e.sx, e.sy, cell, cell, x, y, sz, sz);
    else { g.fillStyle = "#2b2f38"; g.fillRect(x, y, sz, sz); }
    g.strokeStyle = "rgba(0,0,0,.5)"; g.lineWidth = 1;
    g.strokeRect(x + 0.5, y + 0.5, sz - 1, sz - 1);
  }
}

function initAxis() {
  buildEndRows();
  document.getElementById("axis-build").addEventListener("click", buildAxis);
  document.getElementById("axis-clear").addEventListener("click", clearAxis);
  document.getElementById("axis-export").addEventListener("click", async () => {
    if (!axisToken) return;
    const res = await fetch(`/api/export?token=${activeToken()}&axis=${axisToken}`);
    if (!res.ok) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await res.blob());
    a.download = "atlas-axis.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  document.getElementById("axis-strip-canvas").addEventListener("click", (e) => {
    const spec = axisPayload && axisPayload.spectrum;
    if (!spec || !spec.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    let best = spec[0], bd = Infinity;
    for (const s of spec) { const d = Math.abs(s.pos - px); if (d < bd) { bd = d; best = s; } }
    pinnedId = best.id;
    updateDetail(best.id);
    requestRender();
  });
  document.getElementById("axis-box").classList.remove("hidden");
}

/* ----------------------------------------------------- labels & density UI */

function wireToggle(id, rowId, get, set) {
  const cb = document.getElementById(id);
  cb.checked = get();
  cb.addEventListener("change", () => { set(cb.checked); requestRender(); });
  document.getElementById(rowId).classList.remove("hidden");
}

async function loadLabelsAndDensity() {
  try {
    const data = await (await fetch("/api/labels")).json();
    labels = data.labels || [];
  } catch (e) { labels = []; }
  if (labels.length) {
    maxDataLevel = labels.reduce((m, l) => Math.max(m, l.level), 0);
    wireToggle("labels-toggle", "labels-toggle-row", () => showLabels,
      (v) => { showLabels = v; localStorage.setItem("atlasLabels", v ? "1" : "0"); });
  }
  if (manifest.has_density) {
    const img = new Image();
    img.onload = () => { densityImg = img; requestRender(); };
    img.src = "density.webp";   // relative: works at "/" and under a subpath
    try {
      const c = await (await fetch("/api/contours")).json();
      if (c.levels && c.levels.length) contours = c;
    } catch (e) { /* no contours */ }
    wireToggle("density-toggle", "density-toggle-row", () => showDensity,
      (v) => { showDensity = v; localStorage.setItem("atlasDensity", v ? "1" : "0"); });
  }
  requestRender();
}

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
  if (manifest.has_search) document.getElementById("search-box").classList.remove("hidden");
  if (manifest.has_axis) initAxis();
  loadColumns();
  loadLabelsAndDensity();
  requestRender();
  scheduleFetch(true);
}
init();
