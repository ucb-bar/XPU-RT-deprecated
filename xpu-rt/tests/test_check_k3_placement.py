"""scripts/check_k3_placement.py against hand-written placement rows."""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import check_k3_placement as chk  # noqa: E402

REGISTRY = os.path.join(_REPO, "ModelBlaster", "cores", "spacemit_k3.json")


def row(eid, kind, impl, backend, hart, before, after=None, harts=None):
    return {"entry_id": str(eid), "network": "ffn_block", "instance": "0",
            "dispatch_id": str(eid), "op": "linear_s8", "core_kind": kind, "impl": impl,
            "backend": backend, "hart": str(hart), "harts": harts or str(hart),
            "cpu_before": str(before), "cpu_after": str(before if after is None else after)}


@unittest.skipUnless(os.path.exists(REGISTRY), "ModelBlaster submodule not checked out")
class CheckK3Placement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_cpu = chk.load_registry(REGISTRY)

    def test_a_correct_mixed_run_passes(self):
        rows = [row(0, "a100", "ime2", "ime2_a100", 8, 8), row(1, "x100", "rvv", "rvv_x60", 0, 0),
                row(2, "a100", "a100", "rvv_x60", 9, 9), row(3, "x100", "ime", "ime_x60", 3, 3),
                row(4, "a100", "ime2", "ime2_a100", 10, 10, harts="10;11"),
                row(5, "x100", "ime", "ime_x100", 1, 1)]                       # the K3-only X100 backend
        self.assertEqual(chk.check_rows(rows, self.by_cpu), [])

    def test_each_violation_is_reported(self):
        cases = {
            "never ran": row(0, "a100", "ime2", "ime2_a100", 8, -1, -1),
            "ran on 0": row(0, "a100", "ime2", "ime2_a100", 8, 0),          # unlabelled A100 worker
            "ran on 1 -> 2": row(0, "x100", "rvv", "rvv_x60", 1, 1, 2),   # migrated mid-kernel
            "not legal": row(0, "a100", "ime", "ime_x60", 8, 8),            # X100 tile on an A100
            "span kinds": row(0, "x100", "rvv", "rvv_x60", 7, 7, harts="7;8"),
            "expected ime2_a100": row(0, "a100", "ime2", "ime_x60", 8, 8),   # pairing slip
            "ran backend ime_x100, expected ime2_a100": row(0, "a100", "ime2", "ime_x100", 8, 8),
            "expected rvv_x60": row(0, "a100", "a100", "ime2_a100", 8, 8),
        }
        for want, r in cases.items():
            with self.subTest(want):
                bad = chk.check_rows([r], self.by_cpu)
                self.assertTrue(any(want in b for b in bad), bad)

    def test_view_ops_without_a_kernel_are_skipped(self):
        r = row(0, "x100", "x100", "rvv_x60", 0, -1, -1)
        r["dispatch_id"] = "-1"
        self.assertEqual(chk.check_rows([r], self.by_cpu), [])

    def test_worker_lines(self):
        out = ("xpurt: worker[0] kind=x100 pinned_hart=0 observed_cpu=0 claims_unbound=1\n"
               "xpurt: worker[1] kind=a100 pinned_hart=8 observed_cpu=3 claims_unbound=1\n")
        n, bad = chk.check_workers(out)
        self.assertEqual(n, 2)
        self.assertEqual(len(bad), 1)
        self.assertIn("pinned to cpu 8, observed on cpu 3", bad[0])


if __name__ == "__main__":
    unittest.main()
