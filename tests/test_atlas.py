"""Test suite for Image Atlas. Run with:  python -m unittest discover tests -v"""
from __future__ import annotations

import csv
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
        preview_max=512, max_zoom=6, threshold=4, force=False,
    ))
    return out


class TestUnits(unittest.TestCase):
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
        shutil.rmtree(cls.tmp)

    def test_manifest_and_validation(self):
        m = self.ds.manifest
        self.assertEqual(m["count"], N_IMAGES)
        self.assertEqual(m["coords_source"], "provided")
        self.assertEqual(
            {c["name"] for c in m["metadata_columns"]}, {"color", "idx"}
        )

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

    def test_image_info(self):
        info = self.ds.image_info(3)
        self.assertEqual(info["color"], "yellow")
        self.assertEqual(info["width"], 64)
        self.assertTrue(info["preview_url"].endswith("00000003.webp"))
        self.assertIsNone(self.ds.image_info(N_IMAGES))
        preview = self.ds.root / info["preview_url"].lstrip("/")
        self.assertTrue(preview.is_file())


if __name__ == "__main__":
    unittest.main()
