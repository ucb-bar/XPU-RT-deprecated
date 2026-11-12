#!/usr/bin/env python3
"""Did each dispatch of a K3 run execute where the schedule put it, with a
kernel that core can run?

WHY. On the K3 a wrong placement does not fail on its own. An A100 worker that
was never labelled an AI task stays on an X100 and its timings are reported as
A100 timings; an X100 `ime` kernel on an A100 returns wrong numbers rather than
trapping. The walker's own checks (self-pin + sched_getcpu at worker start,
ingest's impl legality check) make these unlikely; this script is the
after-the-fact evidence, from what the binary recorded while it ran.

INPUTS
  --placement  <sched>_placement.csv  (run_xpurt_k1.sh with MB_XPURT_PLACEMENT=1)
  --registry   ModelBlaster/cores/spacemit_k3.json
  --stdout     optional: the run's stdout, for the per-worker
               `pinned_hart=N observed_cpu=M` lines

CHECKS (each violation is listed; exit 1 if any)
  1. every entry with a kernel ran (cpu_before != -1); dispatch_id < 0 rows
     are view/chunk ops the walker completes without a kernel, so they are skipped
  2. it ran on its hart, before and after the kernel (cpu_before == cpu_after == hart)
  3. that cpu's registry kind is the entry's core_kind
  4. the entry's impl is legal on that cpu (registry `impls`; impl == core_kind
     is the kind's own backend and always legal), and the backend the binary
     selected for that impl is the one the impl means (EXPECTED_BACKEND), so a
     CORE_KINDS/BACKENDS pairing slip cannot run an X100 tile on an A100
  5. a composite entry's helper harts are all of the same kind as its master
  6. (with --stdout) every pinned worker's observed_cpu equals its pinned_hart

Not covered: which cpu each pool helper ran its slice on (the pool does not
record it), so check 5 is about the reserved harts, not where slices landed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter

# What each impl may execute on the K3. Anything else (rvv, x100, a100) is a
# vector impl and must be rvv_x60. ime and ime2 share an encoding with
# different tiles, so a swapped pairing gives wrong answers, not a trap.
# ime is the X100 tile: the K1 kernels (ime_x60) or the K3-only X100 backend
# (ime_x100, Director 2026-10-29); ime2 is the A100 tile.
EXPECTED_BACKEND = {"ime": ("ime_x60", "ime_x100"), "ime2": ("ime2_a100",)}
VECTOR_BACKEND = "rvv_x60"

_WORKER_RE = re.compile(r"xpurt: worker\[(\d+)\] kind=(\S+) pinned_hart=(-?\d+) observed_cpu=(-?\d+)")


def load_registry(path: str) -> dict[int, dict]:
    """cpu -> {"name", "kind", "impls"} from a cores/*.json registry."""
    with open(path) as f:
        doc = json.load(f)
    by_cpu: dict[int, dict] = {}
    for core in doc["cores"]:
        for hart in core.get("harts", []):
            by_cpu[int(hart)] = {"name": core["name"], "kind": core["kind"],
                                 "impls": tuple(core.get("impls", ()))}
    return by_cpu


def check_rows(rows: list[dict], by_cpu: dict[int, dict]) -> list[str]:
    bad: list[str] = []
    for r in rows:
        tag = f"entry {r['entry_id']} ({r['network']}#{r['instance']} d{r['dispatch_id']} {r['op']})"
        if int(r["dispatch_id"]) < 0:
            continue
        hart, before, after = int(r["hart"]), int(r["cpu_before"]), int(r["cpu_after"])
        want_be = EXPECTED_BACKEND.get(r["impl"], (VECTOR_BACKEND,))
        if r["backend"] not in want_be:
            bad.append(f"{tag}: impl {r['impl']} ran backend {r['backend']}, expected {' or '.join(want_be)}")
        if before < 0:
            bad.append(f"{tag}: never ran")
            continue
        if not before == after == hart:
            bad.append(f"{tag}: scheduled on cpu {hart}, ran on {before} -> {after}")
        core = by_cpu.get(before)
        if core is None:
            bad.append(f"{tag}: ran on cpu {before}, which the registry does not list")
            continue
        if core["kind"] != r["core_kind"]:
            bad.append(f"{tag}: core_kind {r['core_kind']} but cpu {before} is {core['kind']}")
        impl = r["impl"]
        if impl != r["core_kind"] and core["impls"] and impl not in core["impls"]:
            bad.append(f"{tag}: impl {impl} ({r['backend']}) is not legal on {core['name']} "
                       f"(legal: {list(core['impls'])})")
        harts = [int(h) for h in r["harts"].split(";") if h] if r.get("harts") else [hart]
        kinds = {by_cpu.get(h, {}).get("kind") for h in harts}
        if len(kinds) != 1:
            bad.append(f"{tag}: reserved harts {harts} span kinds {sorted(map(str, kinds))}")
    return bad


def check_workers(stdout_text: str) -> tuple[int, list[str]]:
    bad, n = [], 0
    for m in _WORKER_RE.finditer(stdout_text):
        n += 1
        w, kind, pinned, observed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if pinned >= 0 and observed != pinned:
            bad.append(f"worker {w} ({kind}): pinned to cpu {pinned}, observed on cpu {observed}")
    return n, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--placement", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--stdout", default=None)
    args = ap.parse_args(argv)

    by_cpu = load_registry(args.registry)
    with open(args.placement) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"FAIL: {args.placement} has no rows")
        return 1
    bad = check_rows(rows, by_cpu)
    n_workers = 0
    if args.stdout:
        with open(args.stdout, errors="replace") as f:
            n_workers, wbad = check_workers(f.read())
        if n_workers == 0:
            wbad = [f"no 'xpurt: worker[...]' lines in {args.stdout}"]
        bad += wbad

    ran = Counter((r["core_kind"], r["impl"], r["backend"], r["cpu_before"]) for r in rows)
    print(f"{len(rows)} entries, {n_workers} workers checked")
    for (kind, impl, backend, cpu), n in sorted(ran.items()):
        print(f"  {kind:6s} impl={impl:6s} backend={backend:10s} cpu={cpu:>3s}  x{n}")
    if bad:
        print(f"FAIL: {len(bad)} placement violation(s)")
        for b in bad[:50]:
            print(f"  - {b}")
        return 1
    print("PASS: every entry ran on its scheduled cpu with a kernel that cpu can run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
