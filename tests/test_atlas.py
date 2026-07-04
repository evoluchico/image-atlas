"""Test suite for Image Atlas. Run with:  python -m unittest discover tests -v"""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import build  # noqa: E402
from atlas.server import Dataset, DatasetError, points_in_polygon  # noqa: E402

N_IMAGES = 40
COLORS = [
    ("red", (200, 40, 40)), ("green", (40, 180, 80)),
    ("blue", (40, 80, 200)), ("yellow", (220, 200, 60)),
]


def build_fixture(root: Path) -> Path:
    """Tiny deterministic dataset: 40 solid-color images, coords on a grid."""
    images = root / "images"
    images.mkdir(parents=True)
    meta_rows, coord_rows = [], []
    for i in range(N_IMAGES):
        name, rgb = COLORS[i % len(COLORS)]
        fname = f"img_{i:03d}.png"
        Image.new("RGB", (64, 48), rgb).save(images / fname)
        meta_rows.append({"filename": fname, "color": name, "idx": i})
        coord_rows.append({"path": fname, "x": (i % 8) / 7.0, "y": (i // 8) / 4.0})
    for fn, rows in (("meta.csv", meta_rows), ("coords.csv", coord_rows)):
        with open(root / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    out = root / "dataset"
    build.run(SimpleNamespace(
        images=str(images), files=None, out=str(out), metadata=str(root / "meta.csv"),
        coords=str(root / "coords.csv"), name="fixture", workers=1, cell=48,
        preview_max=512, max_zoom=6, threshold=4, label_column="color",
        hide_columns="idx", force=False,
    ))
    return out


def add_embeddings(root: Path, dim: int = 4) -> None:
    """Give the bundle synthetic per-color unit embeddings so axis tests need no
    model download: image i's vector is basis e_(i%4) + tiny noise, normalized."""
    rng = np.random.default_rng(0)
    E = np.zeros((N_IMAGES, dim), np.float32)
    for i in range(N_IMAGES):
        E[i, i % len(COLORS)] = 1.0
        E[i] += 0.01 * rng.standard_normal(dim).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    (root / "search").mkdir(exist_ok=True)
    np.save(root / "search" / "embeddings.npy", E.astype(np.float16))
    mpath = root / "manifest.json"
    m = json.loads(mpath.read_text())
    m["search"] = {"model": "dummy", "dim": dim}
    mpath.write_text(json.dumps(m))


class TestUnits(unittest.TestCase):
    def test_compute_tile_indexes_matches_written(self):
        rng = np.random.default_rng(3)
        xy = rng.random((200, 2)).astype(np.float32)
        rep = rng.random(200).astype(np.float32)
        idx = build.compute_tile_indexes(xy, rep, 3)
        with tempfile.TemporaryDirectory() as td:
            build.write_tile_indexes(Path(td), xy, rep, 3)
            for z, (order, keys, starts, reps) in idx.items():
                np.testing.assert_array_equal(order, np.load(Path(td) / f"z{z}_order.npy"))
                np.testing.assert_array_equal(keys, np.load(Path(td) / f"z{z}_keys.npy"))
                np.testing.assert_array_equal(starts, np.load(Path(td) / f"z{z}_starts.npy"))
                np.testing.assert_array_equal(reps, np.load(Path(td) / f"z{z}_rep.npy"))

    def test_tile_index_matches_bruteforce(self):
        rng = np.random.default_rng(7)
        xy = rng.random((500, 2)).astype(np.float32)
        rep = rng.random(500).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            build.write_tile_indexes(d, xy, rep, max_zoom=4)
            for z in (0, 2, 4):
                order = np.load(d / f"z{z}_order.npy")
                keys = np.load(d / f"z{z}_keys.npy")
                starts = np.load(d / f"z{z}_starts.npy")
                tile_reps = np.load(d / f"z{z}_rep.npy")
                side = 1 << z
                tx = np.minimum((xy[:, 0] * side).astype(int), side - 1)
                ty = np.minimum((xy[:, 1] * side).astype(int), side - 1)
                want = ty * side + tx
                self.assertEqual(starts[-1], len(xy))
                for i, k in enumerate(keys):
                    members = order[starts[i]: starts[i + 1]]
                    expect = np.flatnonzero(want == k)
                    np.testing.assert_array_equal(np.sort(members), expect)
                    self.assertIn(tile_reps[i], members)  # rep is a member

    def test_points_in_polygon(self):
        square = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
        pts = np.array([[0.5, 0.5], [0.1, 0.5], [0.9, 0.9], [0.21, 0.79]])
        np.testing.assert_array_equal(
            points_in_polygon(pts, square), [True, False, False, True]
        )
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(
            points_in_polygon(np.array([[0.25, 0.25], [0.9, 0.9]]), tri),
            [True, False],
        )

    def test_type_inference(self):
        self.assertEqual(build._infer_type(["1", "2", ""])[0], "INTEGER")
        self.assertEqual(build._infer_type(["1.5", "2"])[0], "REAL")
        self.assertEqual(build._infer_type(["a", "1"])[0], "TEXT")


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="atlas-test-"))
        cls.ds = Dataset(build_fixture(cls.tmp))

    @classmethod
    def tearDownClass(cls):
        cls.ds.close()   # release memory-maps so Windows can delete the dir
        shutil.rmtree(cls.tmp)

    def test_manifest_and_validation(self):
        m = self.ds.manifest
        self.assertEqual(m["count"], N_IMAGES)
        self.assertEqual(m["coords_source"], "provided")
        self.assertEqual(
            {c["name"] for c in m["metadata_columns"]}, {"color", "idx"}
        )

    def test_labels_and_density(self):
        m = self.ds.manifest
        self.assertTrue(m.get("has_density"))
        self.assertEqual(m.get("labels_column"), "color")
        self.assertTrue((self.ds.root / "density.webp").is_file())
        contours = json.loads((self.ds.root / "density_contours.json").read_text())
        self.assertTrue(contours["levels"])
        for lvl in contours["levels"]:                      # segments are 4-tuples
            self.assertEqual(len(lvl["segments"]) % 4, 0)
        data = json.loads((self.ds.root / "labels.json").read_text())
        names = {lab["text"] for lab in data["labels"]}
        self.assertEqual(names, {name for name, _ in COLORS})   # one per color
        for lab in data["labels"]:
            self.assertTrue(0 <= lab["x"] <= 1 and 0 <= lab["y"] <= 1)
            self.assertGreaterEqual(lab["level"], 0)

    def test_broken_bundle_rejected(self):
        broken = self.tmp / "broken"
        shutil.copytree(self.ds.root, broken)
        (broken / "points" / "rep.npy").unlink()
        with self.assertRaises(DatasetError):
            Dataset(broken)
        shutil.rmtree(broken)

    def test_sprite_store_and_strip(self):
        c = self.ds.cell
        for img_id in (0, 1, 5, N_IMAGES - 1):
            want = np.array(COLORS[img_id % len(COLORS)][1])
            got = self.ds.sprites[img_id, c // 2, c // 2].astype(int)
            self.assertLess(np.abs(got - want).max(), 20, f"id={img_id}")
        # strip: requested sprites appear at index-derived cells
        import io
        ids = [3, 0, 7, 1, 2]
        strip = np.asarray(Image.open(io.BytesIO(self.ds.sprite_strip(ids))))
        for j, img_id in enumerate(ids):
            want = np.array(COLORS[img_id % len(COLORS)][1])
            got = strip[c // 2, j * c + c // 2].astype(int)
            self.assertLess(np.abs(got - want).max(), 25, f"strip pos {j}")

    def test_filter_and_counts(self):
        token, count = self.ds.make_filter("color = 'red'")
        self.assertEqual(count, N_IMAGES // 4)
        token2, count2 = self.ds.make_filter("color = 'red'")
        self.assertEqual(token, token2)  # cache hit
        _, c_all = self.ds.make_filter("")
        self.assertEqual(c_all, N_IMAGES)
        with self.assertRaises(sqlite3.Error):
            self.ds.make_filter("nope = 1")

    def test_viewport_conservation(self):
        token, count = self.ds.make_filter("color = 'blue' OR idx < 6")
        for z in range(0, 7):
            for tok, want in ((token, count), ("all", N_IMAGES)):
                r = self.ds.viewport(z, 0, 0, 1, 1, tok)
                total = sum(a["count"] for a in r["aggregates"]) + len(r["items"])
                self.assertEqual(total, want, f"z={z} token={tok}")
        # aggregates appear only above threshold; representative is a survivor
        mask = self.ds.get_mask(token)
        for a in self.ds.viewport(0, 0, 0, 1, 1, token)["aggregates"]:
            self.assertGreater(a["count"], self.ds.threshold)
            self.assertTrue(mask[a["id"]])

    def test_columns_and_structured_filter(self):
        cols = {c["name"]: c for c in self.ds.columns_summary()}
        self.assertEqual(cols["color"]["kind"], "choice")
        self.assertEqual(set(cols["color"]["values"]), {n for n, _ in COLORS})
        self.assertNotIn("idx", cols)   # hidden via filter_hide (fixture builds with it)
        # choice filter
        _, count = self.ds.make_filter_structured([{"col": "color", "values": ["red"]}])
        self.assertEqual(count, N_IMAGES // 4)
        # a hidden column is still filterable (e.g. structured, or Advanced SQL)
        _, c2 = self.ds.make_filter_structured([{"col": "idx", "min": 0, "max": 9}])
        self.assertEqual(c2, 10)
        # combined, and unknown columns ignored safely
        _, c3 = self.ds.make_filter_structured(
            [{"col": "color", "values": ["red", "blue"]}, {"col": "nope", "values": ["x"]}]
        )
        self.assertEqual(c3, N_IMAGES // 2)
        self.assertEqual(self.ds.make_filter_structured([])[0], "all")

    def test_viewport_unknown_token(self):
        with self.assertRaises(KeyError):
            self.ds.viewport(2, 0, 0, 1, 1, "bogus")

    def test_selection_and_export(self):
        whole = [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]]
        token, count = self.ds.make_selection(whole, "all")
        self.assertEqual(count, N_IMAGES)
        ftok, fcount = self.ds.make_filter("color = 'green'")
        stok, scount = self.ds.make_selection(whole, ftok)
        self.assertEqual(scount, fcount)  # lasso ∩ filter
        body = self.ds.export_csv(stok).decode()
        rows = body.strip().splitlines()
        self.assertEqual(len(rows) - 1, scount)
        self.assertIn("color", rows[0])
        self.assertIn(",x,y", rows[0])
        with self.assertRaises(ValueError):
            self.ds.make_selection([[0, 0], [1, 1]], "all")

    def test_tile_members(self):
        token, count = self.ds.make_filter("color = 'red'")
        agg = self.ds.viewport(0, 0, 0, 1, 1, token)["aggregates"][0]
        r = self.ds.tile_members(0, agg["tx"], agg["ty"], token, 0, 500)
        self.assertEqual(r["total"], count)        # one z0 tile holds everything
        self.assertEqual(len(r["ids"]), count)
        mask = self.ds.get_mask(token)
        self.assertTrue(all(mask[i] for i in r["ids"]))
        reps = [float(self.ds.rep[i]) for i in r["ids"]]
        self.assertEqual(reps, sorted(reps, reverse=True))  # best first
        page = self.ds.tile_members(0, agg["tx"], agg["ty"], token, 2, 3)
        self.assertEqual(page["ids"], r["ids"][2:5])        # pagination
        empty = self.ds.tile_members(6, 1, 1, token, 0, 10)  # no fixture point here
        self.assertEqual(empty, {"total": 0, "ids": []})

    def test_ocr_search(self):
        # add an OCR FTS table to a copy and search it (CLIP-free serve path)
        copy = self.tmp / "ocr"
        shutil.copytree(self.ds.root, copy)
        con = sqlite3.connect(copy / "metadata.sqlite")
        con.execute("CREATE VIRTUAL TABLE ocr USING fts5(text)")
        con.execute("INSERT INTO ocr (rowid, text) VALUES (?, ?)", (5, "hello vaccine world"))
        con.execute("INSERT INTO ocr (rowid, text) VALUES (?, ?)", (9, "vaccine news"))
        con.execute("INSERT INTO ocr (rowid, text) VALUES (?, ?)", (12, "unrelated text"))
        con.commit(); con.close()
        with Dataset(copy) as ds:
            self.assertTrue(ds.has_ocr and ds.has_search)
            self.assertEqual(sorted(ds._ocr_ids("vaccine", 10)), [5, 9])
            # exact (text) mode: true count + ranked ids returned
            token, count, ids, exact = ds.search("vaccine", mode="text")
            self.assertEqual(count, 2)
            self.assertTrue(exact)
            self.assertEqual(sorted(ids), [5, 9])
            mask = ds.get_mask(token)
            self.assertTrue(mask[5] and mask[9] and not mask[12])
            # composes with a base filter
            base, _ = ds.make_filter_structured([{"col": "idx", "min": 0, "max": 7}])
            _, c2, ids2, _ = ds.search("vaccine", base, mode="text")
            self.assertEqual(c2, 1)   # only id 5 is within idx<=7
            self.assertEqual(ids2, [5])
        shutil.rmtree(copy)

    def test_retile(self):
        from atlas import retile
        copy = self.tmp / "retiled"
        shutil.copytree(self.ds.root, copy)
        retile.run(SimpleNamespace(dataset=str(copy), max_zoom=3, threshold=2))
        with Dataset(copy) as ds2:
            self.assertEqual(ds2.zmax, 3)
            self.assertEqual(ds2.threshold, 2)
            self.assertFalse((copy / "points" / "z4_order.npy").exists())  # stale levels gone
            r = ds2.viewport(3, 0, 0, 1, 1, "all")
            total = sum(a["count"] for a in r["aggregates"]) + len(r["items"])
            self.assertEqual(total, N_IMAGES)
        shutil.rmtree(copy)

    def test_image_info(self):
        info = self.ds.image_info(3)
        self.assertEqual(info["color"], "yellow")
        self.assertEqual(info["width"], 64)
        self.assertTrue(info["preview_url"].endswith("00000003.webp"))
        self.assertIsNone(self.ds.image_info(N_IMAGES))
        preview = self.ds.root / info["preview_url"].lstrip("/")
        self.assertTrue(preview.is_file())


class TestAxis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="atlas-axis-"))
        root = cls.tmp / "axis"
        shutil.copytree(build_fixture(cls.tmp), root)
        add_embeddings(root)
        cls.ds = Dataset(root)
        cls.red = [i for i in range(N_IMAGES) if i % 4 == 0]
        cls.green = [i for i in range(N_IMAGES) if i % 4 == 1]
        cls.blue = [i for i in range(N_IMAGES) if i % 4 == 2]
        cls.yellow = [i for i in range(N_IMAGES) if i % 4 == 3]

    @classmethod
    def tearDownClass(cls):
        cls.ds.close()
        shutil.rmtree(cls.tmp)

    def test_has_axis(self):
        self.assertTrue(self.ds.has_axis)

    def test_overlay_scores_and_tint(self):
        tok, rec = self.ds.make_axis(
            {"a": {"ids": self.red}, "b": {"ids": self.blue}}, None, "all")
        self.assertEqual(rec["mode"], "overlay")
        norm = rec["x"]["norm"]
        self.assertGreater(float(norm[self.red].mean()), 0.8)    # end A high
        self.assertLess(float(norm[self.blue].mean()), 0.2)      # end B low
        r = self.ds.viewport(6, 0, 0, 1, 1, "all", tok)          # z6: every image an item
        self.assertTrue(r["items"])
        for it in r["items"]:
            self.assertTrue(0.0 <= it["score"] <= 1.0)
            self.assertAlmostEqual(it["score"], float(norm[it["id"]]), places=4)

    def test_union_of_sources_end(self):
        # an end defined by a token (e.g. a lasso/filter group) behaves like its ids
        ftok, _ = self.ds.make_filter("color = 'red'")
        _, rec = self.ds.make_axis(
            {"a": {"tokens": [ftok]}, "b": {"ids": self.blue}}, None, "all")
        self.assertGreater(float(rec["x"]["norm"][self.red].mean()), 0.8)

    def test_scatter_layout_and_quadrants(self):
        tok, rec = self.ds.make_axis(
            {"a": {"ids": self.red}, "b": {"ids": self.blue}},
            {"a": {"ids": self.green}, "b": {"ids": self.yellow}}, "all")
        self.assertEqual(rec["mode"], "scatter")
        axy = rec["layout"]["xy"]
        self.assertEqual(axy.shape, (N_IMAGES, 2))
        # end A (red) → high X (right); end B (blue) → low X (left)
        self.assertGreater(axy[self.red].mean(0)[0], axy[self.blue].mean(0)[0])
        # Y is inverted so end A (green) is at the TOP (small world-y)
        self.assertLess(axy[self.green].mean(0)[1], axy[self.yellow].mean(0)[1])
        r = self.ds.viewport(6, 0, 0, 1, 1, "all", tok)          # served from axis tiles
        self.assertTrue(r["items"])
        for it in r["items"]:
            self.assertTrue(0.0 <= it["x"] <= 1.0 and 0.0 <= it["y"] <= 1.0)
            self.assertNotIn("score", it)                        # no tint in scatter
        # drill-down reads the axis tiling too
        agg = self.ds.viewport(2, 0, 0, 1, 1, "all", tok)["aggregates"]
        if agg:
            m = self.ds.tile_members(2, agg[0]["tx"], agg[0]["ty"], "all", 0, 500, tok)
            self.assertGreater(m["total"], 0)

    def test_export_axis_columns(self):
        stok, rec = self.ds.make_axis(
            {"a": {"ids": self.red}, "b": {"ids": self.blue}},
            {"a": {"ids": self.green}, "b": {"ids": self.yellow}}, "all")
        head = self.ds.export_csv("all", stok).decode().splitlines()
        self.assertIn("axis_x", head[0])
        self.assertIn("axis_y", head[0])
        self.assertIn("quadrant", head[0])
        self.assertEqual(len(head) - 1, N_IMAGES)
        # overlay export carries axis_x but not axis_y
        otok, _ = self.ds.make_axis(
            {"a": {"ids": self.red}, "b": {"ids": self.blue}}, None, "all")
        ohead = self.ds.export_csv("all", otok).decode().splitlines()[0]
        self.assertIn("axis_x", ohead)
        self.assertNotIn("axis_y", ohead)


if __name__ == "__main__":
    unittest.main()
