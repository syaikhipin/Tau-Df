"""R003 — cross-check our E1 re-implementation against the upstream reference.

Runs both implementations back to back on the same data conventions and asserts the
plan's ±0.02 tolerance. Writes experiments/E1/results/r003_crosscheck.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp1 import load_california, load_taiwan, per_channel_credit, select_in_sample_vs_query  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notes/upstream/Learned-Kernel/src"))
from lkbook import load_california as up_california, load_taiwan as up_taiwan  # noqa: E402
from lkbook.chapters import ch07 as up_ch07  # ours vs upstream, same box, same versions  # noqa: E402

OUT = Path(__file__).parent / "results"
TOL = 0.02


def compare_table(ours_rows, up_rows):
    deltas = []
    for ours, up in zip(ours_rows, up_rows):
        for key in ("support_r2", "query_r2", "test_rmse"):
            delta = abs(ours[key] - up[key])
            deltas.append((ours["name"], key, ours[key], up[key], delta))
    return deltas


def main():
    OUT.mkdir(exist_ok=True)
    report = {"tolerance": TOL, "datasets": {}}
    ok = True

    # --- ledger: ours vs upstream, California + Taiwan ---
    for dname, loader_ours, loader_up in [
        ("california", load_california, up_california),
        ("taiwan", load_taiwan, up_taiwan),
    ]:
        d_ours = loader_ours()
        d_up = loader_up()
        ours = per_channel_credit(d_ours)
        up = up_ch07.per_channel_credit(d_up)
        deltas = compare_table(ours, up)
        worst = max(d[4] for d in deltas)
        print(f"\n[{dname}] ledger — worst |ours − upstream| = {worst:.5f}")
        for name, key, o, u, delta in deltas:
            flag = "" if delta <= TOL else "  ← EXCEEDS TOL"
            print(f"  {name:16s} {key:11s} ours={o:.4f}  up={u:.4f}  Δ={delta:.5f}{flag}")
            ok &= delta <= TOL
        report["datasets"][dname] = {
            "ledger_ours": ours, "ledger_upstream": up,
            "max_delta": float(worst), "within_tol": bool(worst <= TOL)}

    # --- fusion selection: ours vs upstream, California ---
    d_ours = load_california()
    d_up = up_california()
    ours = select_in_sample_vs_query(d_ours)
    up = up_ch07.select_in_sample_vs_query(d_up)
    print("\n[fusion] weights / lam / test RMSE, ours vs upstream")
    fusion_deltas = []
    for tag in ("in_sample", "query"):
        for n_ours, n_up in zip(ours["names"], up["names"]):
            o, u = ours[tag]["weights"][n_ours], up[tag]["weights"][n_up]
            delta = abs(o - u)
            flag = "" if delta <= TOL else "  ← EXCEEDS TOL"
            print(f"  {tag:10s} {n_ours:16s} ours={o:.3f}  up={u:.3f}  Δ={delta:.4f}{flag}")
            fusion_deltas.append((tag, n_ours, o, u, delta))
            ok &= delta <= TOL
        o, u = ours[tag]["test_rmse"], up[tag]["test_rmse"]
        delta = abs(o - u)
        print(f"  {tag:10s} {'test RMSE':16s} ours={o:.4f}  up={u:.4f}  Δ={delta:.5f}")
        fusion_deltas.append((tag, "test_rmse", o, u, delta))
        ok &= delta <= TOL
    print(f"  sigma2 est  ours={ours['sigma2']:.5f}  up={up['sigma2']:.5f}")

    report["datasets"]["california"]["fusion_ours"] = ours
    report["datasets"]["california"]["fusion_upstream"] = up
    report["fusion_max_delta"] = float(max(d[4] for d in fusion_deltas))
    report["within_tolerance"] = bool(ok)

    (OUT / "r003_crosscheck.json").write_text(json.dumps(
        report, indent=2, default=float))
    print(f"\n{'PASS' if ok else 'FAIL'} — report: {OUT / 'r003_crosscheck.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
