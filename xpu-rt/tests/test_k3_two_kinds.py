"""K3: two core kinds, profiled as two targets, in one schedule.

The SpacemiT K3 has X100 cores (cpus 0-7, VLEN 256, `ime` = the K1's 4x8x4
matrix tile) and A100 cores (cpus 8-15, VLEN 1024, `ime2` = an 8x16x8 tile).
Both run the same `rvv_x60` kernels, at different speeds. Three things went
wrong when the K1 path was pointed at the K3, and these tests pin each one:

* **One profile target.** With a single `profile.target`, the A100's rvv cell
  read the X100's rvv_x60 profile: A100 work was priced with X100 data. Per-kind
  `profile.targets` turn each combination's hw into a `hw@target` label, so the
  two rvv_x60 measurements are separate keys.

* **Count-only topo tags.** `topo_tag_for_combination` numbers cpus from 0 in
  every kind, so the first A100 read `topo_0` -- an X100 measurement directory.
  `profile.core_ids` tag groups with the kind's real cpus (`topo_8`), anchored
  to the kind's first n cpus the way the K1 already anchors cluster 1.

* **The `rvv -> ime` folder swap.** It cannot name `ime2_a100`, and K1_CAPABILITIES
  has no `ime2`, so the A100 matrix engine could not be offered at all.
  `profile.impls` gives each kind its implementations and their profile folders;
  that set is also the kind's capability set, so `ime` (wrong answers on an
  A100) is never offered there and `ime2` (traps on an X100) never on an X100.

None of the new fields exists in a K1 config, and the K1 schedules are
unchanged (checked by diffing networks_k1_ffn_ime / networks_k1_multicore_shard
schedules before and after).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from capabilities import build_machine_combinations_with_impls  # noqa: E402
from profile_loader import (  # noqa: E402
    _requested_pairs,
    _resolve_topo_for,
    load_profiled_processing_times,
    split_hw_label,
)
from workload_factory import topo_tag_for_cores  # noqa: E402

SCRIPT = os.path.join(_REPO, "scripts", "run_xpurt_schedule.py")
FFN_GRAPH = ("gen_mb/vmfb/ffn_block/spacemit_x60/rvv_x60/ffn_block.int8/"
             "ffn_block.int8_dispatch_graph.json")
CORE_IDS = {"CPU_P": list(range(0, 8)), "CPU_E": list(range(8, 16))}
TARGETS = {"CPU_P": "spacemit_x100", "CPU_E": "spacemit_a100"}
IMPLS = {"CPU_P": {"rvv": "rvv_x60", "ime": "ime_x60"},
         "CPU_E": {"rvv": "rvv_x60", "ime2": "ime2_a100"}}
INFEASIBLE = 1e6  # anything at or above this is "cannot run here"

# Synthetic times (ms) per dispatch of ffn_block: layernorm, fc1, gelu, fc2, add.
OPS = ["layernorm_s8", "linear_s8", "gelu_s8", "linear_s8", "add_s8"]
TREE = {
    ("rvv_x60", "spacemit_x100", "topo_0"): ([1.0] * 5, None),
    ("rvv_x60", "spacemit_a100", "topo_8"): ([2.0] * 5, None),
    ("ime_x60", "spacemit_x100", "topo_0"): ([1.0, 0.5, 1.0, 0.5, 1.0], "ime"),
    ("ime2_a100", "spacemit_a100", "topo_8"): ([2.0, 0.1, 2.0, 0.1, 2.0], "ime2_a100"),
    # Decoys at the paths the old code would read. Any of these leaking into a
    # cost cell shows up as 0.01 ms.
    ("rvv_x60", "spacemit_x100", "topo_8"): ([0.01] * 5, None),
    ("rvv_x60", "spacemit_a100", "topo_0"): ([0.01] * 5, None),
    ("ime2_a100", "spacemit_x100", "topo_0"): ([0.01] * 5, "ime2_a100"),
    ("ime_x60", "spacemit_a100", "topo_8"): ([0.01] * 5, "ime"),
}
HDR = ("dispatch_id,module_name,vmfb_path,mlir_path,mean_time,mean_unit,mean_time_ns,"
       "returncode,log_path,source,op,shape,cycles,cycles_n,cycles_cv_pct,implementation")


def write_tree(gen_root: str) -> None:
    for (hw, target, topo), (times, ime_impl) in TREE.items():
        d = os.path.join(gen_root, "profile", hw, target, "ffn_block", "ffn_block.int8",
                         f"ffn_block_{target}_{hw}_ffn_block.int8", topo)
        os.makedirs(d, exist_ok=True)
        rows = [HDR]
        for i, (op, t) in enumerate(zip(OPS, times)):
            impl = (f"curated[{ime_impl}]/x" if ime_impl and op == "linear_s8"
                    else "curated[rvv]/direct")
            rows.append(f"{i},ffn_block$dispatch_{i}_{hw}_{op},,,{t},ms,{t * 1e6},0,,k1,"
                        f"{op},,1,49,0.1,{impl}")
        with open(os.path.join(d, "results.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")


def k3_combinations(n_p: int = 2, n_e: int = 2):
    """What schedule_iree_networks builds for the K3 config."""
    _, combos, impls = build_machine_combinations_with_impls(
        {"CPU_P": n_p, "CPU_E": n_e}, {k: list(v) for k, v in IMPLS.items()},
        {k: frozenset(v) for k, v in IMPLS.items()}, granularity="per_core")
    labels = [f"{IMPLS[c[0].split('#')[0]][i]}@{TARGETS[c[0].split('#')[0]]}"
              for c, i in zip(combos, impls)]
    return combos, impls, labels


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_xpurt_schedule_k3test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TopoTags(unittest.TestCase):
    def test_a100_groups_are_tagged_with_their_real_cpus(self):
        self.assertEqual(topo_tag_for_cores(["CPU_E#0"], CORE_IDS), "topo_8")
        self.assertEqual(topo_tag_for_cores(["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3"],
                                            CORE_IDS), "topo_8_9_10_11")
        self.assertEqual(topo_tag_for_cores(["CPU_P#0"], CORE_IDS), "topo_0")

    def test_tags_are_anchored_to_the_kinds_first_cpus(self):
        """Decision 3: cpus 10-11 reuse the 8-9 measurement, 4-7 reuse 0-3."""
        self.assertEqual(topo_tag_for_cores(["CPU_E#2", "CPU_E#3"], CORE_IDS), "topo_8_9")
        self.assertEqual(topo_tag_for_cores(["CPU_P#5"], CORE_IDS), "topo_0")

    def test_bad_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            topo_tag_for_cores(["CPU_P#0", "CPU_E#0"], CORE_IDS)
        with self.assertRaises(KeyError):
            topo_tag_for_cores(["GPU#0"], CORE_IDS)
        with self.assertRaises(ValueError):
            topo_tag_for_cores([f"CPU_E#{i}" for i in range(9)], CORE_IDS)

    def test_without_core_ids_the_k1_tag_is_unchanged(self):
        self.assertEqual(_resolve_topo_for("rvv_x60", ["CPU_E#0", "CPU_E#1"], None), "topo_0_1")
        self.assertEqual(_resolve_topo_for("rvv_x60", ["CPU_E#0"], {"ime_x60": "topo_9"}),
                         "topo_0")
        self.assertEqual(_resolve_topo_for("rvv_x60", ["CPU_E#0"], "topo_0_1_2_3"),
                         "topo_0_1_2_3")

    def test_without_core_ids_the_k1_pairs_are_the_full_cross_product(self):
        """The K1 path asks for every hw at every combination's tag, as before."""
        combos = [["CPU_P#0"], ["CPU_P#0", "CPU_P#1"], ["CPU_E#0"]]
        pairs = _requested_pairs(["rvv_x60", "ime_x60", "rvv_x60"], combos, None, None)
        self.assertEqual(pairs, {(hw, t) for hw in ("rvv_x60", "ime_x60")
                                 for t in ("topo_0", "topo_0_1")})

    def test_hw_labels_split_into_folder_and_target(self):
        self.assertEqual(split_hw_label("ime2_a100@spacemit_a100", "spacemit_x60"),
                         ("ime2_a100", "spacemit_a100"))
        self.assertEqual(split_hw_label("rvv_x60", "spacemit_x60"), ("rvv_x60", "spacemit_x60"))

    def test_each_kind_requests_only_its_own_profiles(self):
        combos, _, labels = k3_combinations()
        pairs = _requested_pairs(labels, combos, None, CORE_IDS)
        self.assertEqual(pairs, {
            ("rvv_x60@spacemit_x100", "topo_0"), ("ime_x60@spacemit_x100", "topo_0"),
            ("rvv_x60@spacemit_a100", "topo_8"), ("ime2_a100@spacemit_a100", "topo_8")})


class ImplementationsPerKind(unittest.TestCase):
    def test_ime2_only_on_a100_and_ime_only_on_x100(self):
        combos, impls, _ = k3_combinations(n_p=8, n_e=8)
        for combo, impl in zip(combos, impls):
            kind = combo[0].split("#")[0]
            if impl == "ime2":
                self.assertEqual(kind, "CPU_E", combo)
            if impl == "ime":
                self.assertEqual(kind, "CPU_P", combo)
        self.assertEqual(sorted(set(impls)), ["ime", "ime2", "rvv"])


class ProfileKindsConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()

    def parse(self, profile, machines=None):
        return self.mod._parse_profile_kinds(profile, machines or {"CPU_P": 2, "CPU_E": 2})

    def full(self):
        return {"targets": {"cpu_p": "spacemit_x100", "cpu_e": "spacemit_a100"},
                "impls": {"cpu_p": dict(IMPLS["CPU_P"]), "cpu_e": dict(IMPLS["CPU_E"])},
                "core_ids": {"cpu_p": list(CORE_IDS["CPU_P"]), "cpu_e": list(CORE_IDS["CPU_E"])}}

    def test_a_k1_profile_block_yields_nothing(self):
        out = self.parse({"target": "spacemit_x60", "topo_tag": "topo_0"})
        self.assertEqual(out, {"profile_targets": None, "profile_impls": None,
                               "profile_core_ids": None})

    def test_the_k3_block_is_read_with_upper_case_kinds(self):
        out = self.parse(self.full())
        self.assertEqual(out["profile_targets"], TARGETS)
        self.assertEqual(out["profile_impls"], IMPLS)
        self.assertEqual(out["profile_core_ids"], CORE_IDS)

    def test_fields_must_come_together(self):
        cfg = self.full()
        del cfg["core_ids"]
        with self.assertRaises(ValueError):
            self.parse(cfg)

    def test_every_machine_kind_needs_an_entry(self):
        cfg = self.full()
        del cfg["targets"]["cpu_e"]
        with self.assertRaises(ValueError):
            self.parse(cfg)

    def test_a_cpu_cannot_belong_to_two_kinds(self):
        cfg = self.full()
        cfg["core_ids"]["cpu_e"] = [7, 8, 9]
        with self.assertRaises(ValueError):
            self.parse(cfg)

    def test_fewer_cpu_ids_than_machines_is_rejected(self):
        cfg = self.full()
        cfg["core_ids"]["cpu_e"] = [8]
        with self.assertRaises(ValueError):
            self.parse(cfg)

    def test_a_topo_override_cannot_be_combined_with_core_ids(self):
        """Copying the X100-only block (topo_tag_override: topo_0) into a
        two-kind config would make the A100 read topo_0."""
        for extra in ({"topo_tag": "topo_0", "topo_tag_override": True},
                      {"topo_tag_per_hw": {"rvv_x60": "topo_0"}}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.parse({**self.full(), **extra})

    def test_malformed_values_are_rejected(self):
        bad = [
            ("targets", lambda c: c.update(targets=["spacemit_x100"])),
            ("target None", lambda c: c["targets"].update(cpu_e=None)),
            ("@ in target", lambda c: c["targets"].update(cpu_e="a@b")),
            ("/ in folder", lambda c: c["impls"]["cpu_e"].update(ime2="x/ime2")),
            ("ime impl, rvv folder", lambda c: c["impls"]["cpu_e"].update(ime2="a100_matrix")),
            ("rvv impl, ime folder", lambda c: c["impls"]["cpu_e"].update(rvv="ime_x60")),
            ("bool cpu id", lambda c: c["core_ids"].update(cpu_e=[True, 9])),
            ("duplicate cpu id", lambda c: c["core_ids"].update(cpu_e=[8, 8])),
            ("unknown kind", lambda c: c["targets"].update(gpu="x")),
        ]
        for name, mutate in bad:
            cfg = self.full()
            mutate(cfg)
            with self.subTest(name), self.assertRaises(ValueError):
                self.parse(cfg)

    def test_a_zero_count_kind_may_stay_listed(self):
        """machines {cpu_p: 8, cpu_e: 0} drops CPU_E; its entries are ignored."""
        out = self.parse(self.full(), machines={"CPU_P": 8})
        self.assertEqual(set(out["profile_targets"]), {"CPU_P"})


class CostCellsComeFromTheirOwnKind(unittest.TestCase):
    """Loader level: every cost cell reads its kind's target at its kind's cpus."""

    def test_each_cell_reads_its_own_measurement(self):
        combos, _, labels = k3_combinations()
        net = {"ffn_block": {"id": 0, "identifier": "ffn_block", "dispatch_deps_path": FFN_GRAPH}}
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(tmp)
            times, _, _, _ = load_profiled_processing_times(
                networks=net, repo_base_path=_REPO, machine_combinations=combos,
                combo_hw=labels, profile_target="spacemit_x100", cpu_p_profile_hw="rvv_x60",
                cpu_e_profile_hw="rvv_x60", rng=np.random.default_rng(0), p_core_speedup=1.0,
                gen_root=tmp, core_ids=CORE_IDS)
        # combos: P rvv x2, P ime x2, E rvv x2, E ime2 x2
        layernorm = times["ffn_block_dispatch_0"]
        fc1 = times["ffn_block_dispatch_1"]
        self.assertEqual(layernorm[:2], [1.0, 1.0])       # X100 rvv
        self.assertEqual(layernorm[4:6], [2.0, 2.0])      # A100 rvv, not the X100 number
        self.assertTrue(all(t >= INFEASIBLE for t in layernorm[2:4] + layernorm[6:8]),
                        "an rvv row in a matrix profile is not a matrix kernel")
        self.assertEqual(fc1, [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 0.1, 0.1])


class EndToEndSchedule(unittest.TestCase):
    """The scheduler script on a K3 config: placements, impls and metadata."""

    def test_k3_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen_root = os.path.join(tmp, "g")
            write_tree(gen_root)
            cfg = {
                "hardware": {
                    "machines": {"cpu_p": 2, "cpu_e": 2},
                    "profile": {
                        "target": "spacemit_x100", "gen_root": gen_root,
                        "targets": {"cpu_p": "spacemit_x100", "cpu_e": "spacemit_a100"},
                        "impls": {"cpu_p": IMPLS["CPU_P"], "cpu_e": IMPLS["CPU_E"]},
                        "core_ids": {"cpu_p": CORE_IDS["CPU_P"], "cpu_e": CORE_IDS["CPU_E"]}},
                    "p_core_speedup": 1.0},
                "scheduler": {"random_seed": 0, "use_profiled": True,
                              "machine_combination_mode": "singletons", "enable_impls": True,
                              "prune_periodic": True},
                "networks": {"ffn_block": {"id": 0, "identifier": "ffn_block",
                                           "dispatch_deps_path": FFN_GRAPH, "period": 100.0,
                                           "window_duration": 100.0, "num_instances": 1}},
                "edges": []}
            cfg_path = os.path.join(tmp, "networks_k3_two_kinds_test.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            run = subprocess.run(
                [sys.executable, SCRIPT, "--networks-json", cfg_path, "--solver", "greedy",
                 "--random-seed", "0"], cwd=tmp, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-3000:])
            with open(os.path.join(tmp, "schedules",
                                   "scheduled_networks_k3_two_kinds_test_greedy_profiled.json")) as f:
                sched = json.load(f)

        entries = sched["dispatches"]
        for key, e in entries.items():
            kind = e["hardware_target"].split("#")[0]
            if e["impl"] == "ime2":
                self.assertEqual(kind, "CPU_E", key)
            if e["impl"] == "ime":
                self.assertEqual(kind, "CPU_P", key)
        by_id = {k.rsplit("_", 1)[-1]: e for k, e in entries.items()}
        for linear in ("1", "3"):   # fc1, fc2: A100 matrix engine, 0.1 ms
            self.assertEqual((by_id[linear]["impl"], by_id[linear]["duration"]), ("ime2", 0.1))
        for other in ("0", "2", "4"):   # X100 rvv, 1.0 ms (A100 rvv is 2.0)
            self.assertEqual(by_id[other]["hardware_target"].split("#")[0], "CPU_P")
            self.assertEqual(by_id[other]["duration"], 1.0)

        meta = sched["metadata"]["profile_kinds"]
        self.assertEqual(meta["core_ids"], CORE_IDS)
        self.assertEqual(meta["hw_labels"], sorted(
            ["rvv_x60@spacemit_x100", "ime_x60@spacemit_x100",
             "rvv_x60@spacemit_a100", "ime2_a100@spacemit_a100"]))
        used = sched["metadata"]["pdb_files"]
        self.assertEqual(len(used), 4)
        for p in used:
            want = "/topo_8/" if "/spacemit_a100/" in p else "/topo_0/"
            self.assertIn(want, p)


if __name__ == "__main__":
    unittest.main()
