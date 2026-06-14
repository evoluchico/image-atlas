/* Static-demo shim: answers the /api/* routes client-side so app.js can run
 * without a backend. Loaded only in exported demos (before app.js); the
 * installed tool never uses this file. Algorithms mirror atlas/server.py at
 * demo scale (a few thousand points), where O(N) per query is negligible.
 */
"use strict";

(() => {
  const realFetch = window.fetch.bind(window);
  let D = null;        // parsed data.json
  let ready = null;    // data.json loaded (gates every route)
  const tokens = new Map([["all", null]]);   // token -> mask (Uint8Array) | null
  const LAMBDA = 0.25;

  function load() {
    if (!ready) {
      ready = (async () => {
        D = await (await realFetch("./data.json")).json();
        addBadge();
      })();
    }
    return ready;
  }

  function addBadge() {
    const a = document.createElement("a");
    a.textContent = "static demo — get Image Atlas";
    a.href = "https://github.com/evoluchico/image-atlas";
    a.style.cssText =
      "position:fixed;left:10px;top:8px;z-index:10;font:12px system-ui;" +
      "color:#9ecbf0;background:rgba(21,23,28,.8);padding:3px 10px;" +
      "border:1px solid #30343f;border-radius:6px;text-decoration:none";
    document.body.appendChild(a);
  }

  const json = (obj, status = 200) =>
    new Response(JSON.stringify(obj), {
      status, headers: { "Content-Type": "application/json" },
    });

  /* ---------------- WHERE parser (subset: COND [AND COND]*) ------------- */

  function compileWhere(where) {
    const colIdx = {};
    D.columns.forEach((c, i) => (colIdx[c.toLowerCase()] = i));
    const preds = [];
    for (const part of where.split(/\s+AND\s+/i)) {
      const m = part.trim().match(
        /^(\w+)\s*(=|!=|<>|<=|>=|<|>|LIKE)\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)$/i
      );
      if (!m) {
        throw new Error(
          `cannot parse "${part.trim()}" — the demo supports: col = 'x', ` +
          `col != / < / <= / > / >= number, col LIKE '%x%', joined by AND`
        );
      }
      const [, col, opRaw, valRaw] = m;
      const i = colIdx[col.toLowerCase()];
      if (i === undefined) throw new Error(`no such column: ${col}`);
      const op = opRaw.toUpperCase();
      const val = valRaw.startsWith("'")
        ? valRaw.slice(1, -1).replace(/''/g, "'")
        : parseFloat(valRaw);
      if (op === "LIKE") {
        const re = new RegExp(
          "^" + String(val).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
            .replace(/%/g, ".*").replace(/_/g, ".") + "$", "i");
        preds.push((row) => row[i] !== null && re.test(String(row[i])));
      } else {
        preds.push((row) => {
          const v = row[i];
          if (v === null) return false;
          const a = typeof val === "number" ? Number(v) : String(v);
          switch (op) {
            case "=": return a === val || String(v) === String(val);
            case "!=": case "<>": return a !== val;
            case "<": return a < val;
            case "<=": return a <= val;
            case ">": return a > val;
            case ">=": return a >= val;
          }
        });
      }
    }
    const mask = new Uint8Array(D.rows.length);
    let count = 0;
    D.rows.forEach((row, r) => {
      if (preds.every((p) => p(row))) { mask[r] = 1; count++; }
    });
    return { mask, count };
  }

  /* ---------------- structured filters (mirror columns_summary) --------- */

  function columnsSummary() {
    const out = [];
    D.columns.forEach((name, i) => {
      const vals = D.rows.map((r) => r[i]).filter((v) => v !== null && v !== "");
      if (!vals.length) return;
      if (vals.every((v) => typeof v === "number")) {
        out.push({ name, kind: "range", min: Math.min(...vals), max: Math.max(...vals) });
      } else {
        const uniq = [...new Set(vals.map(String))].sort();
        out.push(uniq.length <= 60
          ? { name, kind: "choice", values: uniq }
          : { name, kind: "text" });
      }
    });
    return out;
  }

  function structuredMask(filters) {
    const idx = {};
    D.columns.forEach((c, i) => (idx[c] = i));
    const mask = new Uint8Array(D.rows.length);
    let count = 0;
    D.rows.forEach((row, r) => {
      const ok = (filters || []).every((f) => {
        const i = idx[f.col];
        if (i === undefined) return true;
        const v = row[i];
        if (f.values && f.values.length && !f.values.includes(String(v))) return false;
        if (f.min != null && !(Number(v) >= f.min)) return false;
        if (f.max != null && !(Number(v) <= f.max)) return false;
        return true;
      });
      if (ok) { mask[r] = 1; count++; }
    });
    return { mask, count };
  }

  /* --------------- viewport summarization (mirrors server.py) ----------- */

  function tileBuckets(z, x0, y0, x1, y1, mask) {
    const side = 1 << z;
    const tx0 = Math.max(0, Math.min(side - 1, Math.floor(x0 * side)));
    const tx1 = Math.max(0, Math.min(side - 1, Math.floor(x1 * side)));
    const ty0 = Math.max(0, Math.min(side - 1, Math.floor(y0 * side)));
    const ty1 = Math.max(0, Math.min(side - 1, Math.floor(y1 * side)));
    const buckets = new Map();
    for (let i = 0; i < D.x.length; i++) {
      if (mask && !mask[i]) continue;
      const tx = Math.min(side - 1, Math.floor(D.x[i] * side));
      const ty = Math.min(side - 1, Math.floor(D.y[i] * side));
      if (tx < tx0 || tx > tx1 || ty < ty0 || ty > ty1) continue;
      const key = ty * side + tx;
      let b = buckets.get(key);
      if (!b) buckets.set(key, (b = []));
      b.push(i);
    }
    return buckets;
  }

  function viewport(z, x0, y0, x1, y1, mask) {
    const zm = D.manifest.zoom;
    z = Math.max(zm.min, Math.min(zm.max, z));
    const side = 1 << z;
    const thr = D.manifest.aggregate_threshold;
    const aggregates = [], items = [];
    for (const [key, ids] of tileBuckets(z, x0, y0, x1, y1, mask)) {
      if (ids.length <= thr) {
        ids.forEach((i) => items.push({ id: i, x: D.x[i], y: D.y[i] }));
        continue;
      }
      let wsum = 0, cx = 0, cy = 0;
      ids.forEach((i) => {
        const w = D.rep[i];
        wsum += w; cx += D.x[i] * w; cy += D.y[i] * w;
      });
      if (wsum <= 0) { wsum = ids.length; cx = cy = 0;
        ids.forEach((i) => { cx += D.x[i]; cy += D.y[i]; }); }
      cx /= wsum; cy /= wsum;
      let anchor = ids[0], best = -Infinity;
      ids.forEach((i) => {
        const d = (D.x[i] - cx) ** 2 + (D.y[i] - cy) ** 2;
        if (-d > best) { best = -d; anchor = i; }
      });
      const ax = D.x[anchor], ay = D.y[anchor];
      let pick = ids[0]; best = -Infinity;
      ids.forEach((i) => {
        const d = Math.hypot(D.x[i] - ax, D.y[i] - ay) * side;
        const s = D.rep[i] - LAMBDA * d;
        if (s > best) { best = s; pick = i; }
      });
      aggregates.push({ tx: key % side, ty: Math.floor(key / side),
                        count: ids.length, id: pick, x: D.x[pick], y: D.y[pick] });
    }
    return { z, aggregates, items };
  }

  /* ------------------------------ helpers ------------------------------- */

  function previewUrl(id) {
    return `previews/${String(Math.floor(id / 1000)).padStart(3, "0")}/` +
           `${String(id).padStart(8, "0")}.webp`;
  }

  function pointsInPolygon(poly, base) {
    const mask = new Uint8Array(D.x.length);
    let count = 0;
    for (let i = 0; i < D.x.length; i++) {
      if (base && !base[i]) continue;
      const x = D.x[i], y = D.y[i];
      let inside = false;
      for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
        const [xa, ya] = poly[a], [xb, yb] = poly[b];
        if ((ya > y) !== (yb > y) && x < xb + ((y - yb) * (xa - xb)) / (ya - yb)) {
          inside = !inside;
        }
      }
      if (inside) { mask[i] = 1; count++; }
    }
    return { mask, count };
  }

  /* ------------------------------- router ------------------------------- */

  window.fetch = async (resource, opts) => {
    const url = typeof resource === "string" ? resource : resource.url;
    if (!url.startsWith("/api/")) return realFetch(resource, opts);
    await load();
    const u = new URL(url, location.origin);
    const q = u.searchParams;
    const getMask = (tok) => {
      if (!tokens.has(tok)) throw 410;
      return tokens.get(tok);
    };
    try {
      if (u.pathname === "/api/manifest") return json(D.manifest);
      if (u.pathname === "/api/columns") return json({ columns: columnsSummary() });
      if (u.pathname === "/api/labels") return json({ labels: D.labels || [] });
      if (u.pathname === "/api/contours") return json(D.contours || { levels: [] });

      if (u.pathname === "/api/filter") {
        const body = JSON.parse(opts?.body || "{}");
        if ("filters" in body) {
          if (!body.filters.length) return json({ token: "all", count: D.rows.length });
          const { mask, count } = structuredMask(body.filters);
          const token = "s" + Math.random().toString(36).slice(2, 10);
          tokens.set(token, mask);
          return json({ token, count });
        }
        const where = (body.where || "").trim();
        if (!where) return json({ token: "all", count: D.rows.length });
        try {
          const { mask, count } = compileWhere(where);
          const token = "f" + Math.random().toString(36).slice(2, 10);
          tokens.set(token, mask);
          return json({ token, count });
        } catch (e) { return json({ error: e.message }, 400); }
      }

      if (u.pathname === "/api/select") {
        const body = JSON.parse(opts?.body || "{}");
        if (!Array.isArray(body.polygon) || body.polygon.length < 3) {
          return json({ error: "polygon must have at least 3 points" }, 400);
        }
        const base = getMask(body.base_token || "all");
        const { mask, count } = pointsInPolygon(body.polygon, base);
        const token = "sel" + Math.random().toString(36).slice(2, 9);
        tokens.set(token, mask);
        return json({ token, count });
      }

      if (u.pathname === "/api/viewport") {
        return json(viewport(+q.get("z"), +q.get("x0"), +q.get("y0"),
                             +q.get("x1"), +q.get("y1"),
                             getMask(q.get("token") || "all")));
      }

      if (u.pathname === "/api/tile") {
        const mask = getMask(q.get("token") || "all");
        const z = +q.get("z"), tx = +q.get("tx"), ty = +q.get("ty");
        const side = 1 << z;
        const members = [];
        for (let i = 0; i < D.x.length; i++) {
          if (mask && !mask[i]) continue;
          if (Math.min(side - 1, Math.floor(D.x[i] * side)) === tx &&
              Math.min(side - 1, Math.floor(D.y[i] * side)) === ty) members.push(i);
        }
        members.sort((a, b) => D.rep[b] - D.rep[a]);
        const off = +(q.get("offset") || 0), lim = +(q.get("limit") || 60);
        return json({ total: members.length, ids: members.slice(off, off + lim) });
      }

      if (u.pathname === "/api/export") {
        const mask = getMask(q.get("token") || "all");
        const lines = [["id", ...D.columns, "x", "y"].join(",")];
        const quote = (v) => (/[",\n]/.test(String(v ?? "")) ?
          `"${String(v).replace(/"/g, '""')}"` : String(v ?? ""));
        D.rows.forEach((row, i) => {
          if (mask && !mask[i]) return;
          lines.push([i, ...row.map(quote), D.x[i], D.y[i]].join(","));
        });
        return new Response(lines.join("\n"),
          { status: 200, headers: { "Content-Type": "text/csv" } });
      }

      if (u.pathname.startsWith("/api/image/")) {
        const id = +u.pathname.split("/").pop();
        if (!(id >= 0 && id < D.rows.length)) return json({ error: "no such image" }, 404);
        const info = { id };
        D.columns.forEach((c, i) => (info[c] = D.rows[id][i]));
        info.preview_url = previewUrl(id);
        return json(info);
      }

      return json({ error: "not found" }, 404);
    } catch (e) {
      if (e === 410) return json({ error: "unknown token" }, 410);
      throw e;
    }
  };
})();
