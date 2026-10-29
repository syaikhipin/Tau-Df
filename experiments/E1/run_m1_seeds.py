"""M1 — ledger + fusion selection across seeds (runs R004–R007).

E1 pipeline over seeds {0, 1, 2}: per-channel ledger on California + Taiwan,
fusion selection (SURE vs query) on both. Aggregates seed stability and checks the
plan's M1 success criteria. Writes experiments/E1/results/m1_seeds.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from exp1 import load_california, load_taiwan, per_channel_credit, select_in_sample_vs_query  # noqa: E402

OUT = Path(__file__).parent / "results"
SEEDS = [0, 1, 2]


def main():
    OUT.mkdir(exist_ok=True)
    report = {"seeds": SEEDS, "ledger": {}, "fusion": {}}

    for dname, loader in [("california", load_california), ("taiwan", load_taiwan)]:
        d = loader()
        ledgers, fusions = [], []
        for seed in SEEDS:
            print(f"[{dname}] seed {seed}: ledger...", flush=True)
            ledgers.append(per_channel_credit(d, seed=seed))
            print(f"[{dname}] seed {seed}: fusion...", flush=True)
            fusions.append(select_in_sample_vs_query(d, seed=seed))

        # ledger across seeds
        rows = []
        for names_i in range(3):
            name = ledgers[0][names_i]["name"]
            vals = {k: [ledgers[s][names_i][k] for s in range(len(SEEDS))]
                    for k in ("support_r2", "query_r2", "gap", "test_rmse")}
            rows.append({
                "name": name,
                **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
                **{f"{k}_std": float(np.std(v)) for k, v in vals.items()},
            })
        report["ledger"][dname] = rows
        report["ledger"][dname + "_raw"] = ledgers

        print(f"\n[{dname}] ledger across seeds {SEEDS} (mean ± std)")
        for r in rows:
            print(f"  {r['name']:16s} support {r['support_r2_mean']:.3f}±{r['support_r2_std']:.3f}  "
                  f"query {r['query_r2_mean']:.3f}±{r['query_r2_std']:.3f}  "
                  f"gap {r['gap_mean']:.3f}±{r['gap_std']:.3f}")

        # fusion across seeds
        fus = {"in_sample": {"weights": [], "lam": [], "test_rmse": []},
               "query": {"weights": [], "lam": [], "test_rmse": []}}
        names = fusions[0]["names"]
        for f in fusions:
            for tag in fus:
                fus[tag]["weights"].append([f[tag]["weights"][n] for n in names])
                fus[tag]["lam"].append(f[tag]["lam"])
                fus[tag]["test_rmse"].append(f[tag]["test_rmse"])
        report["fusion"][dname] = {
            "names": names,
            "sigma2_first_seed": fusions[0]["sigma2"],
            "in_sample": {k: (np.asarray(v).tolist() if k == "weights" else v) for k, v in fus["in_sample"].items()},
            "query": {k: (np.asarray(v).tolist() if k == "weights" else v) for k, v in fus["query"].items()},
        }
        for tag in ("in_sample", "query"):
            w = np.asarray(fus[tag]["weights"])
            rm = fus[tag]["test_rmse"]
            print(f"  {tag:10s} leaf weight {w[:, 2].mean():.2f}±{w[:, 2].std():.2f}  "
                  f"test RMSE {np.mean(rm):.4f}±{np.std(rm):.4f}")

    # --- M1 success criteria (plan B1/B2) ---
    print("\n--- M1 criteria checks ---")
    checks = {}

    def leaf_gap_largest(dname):
        rows = report["ledger"][dname]
        return rows[2]["gap_mean"] == max(r["gap_mean"] for r in rows)

    checks["ca_leaf_largest_gap"] = leaf_gap_largest("california")
    checks["tw_leaf_largest_gap"] = leaf_gap_largest("taiwan")
    ca_rows = report["ledger"]["california"]
    checks["ca_rbf_gap_gt_ard_gap"] = ca_rows[0]["gap_mean"] > ca_rows[1]["gap_mean"]
    # B2: SURE hands leaf the weight; query demotes it; query RMSE beats in-sample (CA)
    ca_f = report["fusion"]["california"]
    leaf_insample = np.mean([w[2] for w in ca_f["in_sample"]["weights"]])
    leaf_query = np.mean([w[2] for w in ca_f["query"]["weights"]])
    rmse_in = np.mean(ca_f["in_sample"]["test_rmse"])
    rmse_q = np.mean(ca_f["query"]["test_rmse"])
    checks["ca_sure_leaf_weight_ge_0.9"] = bool(leaf_insample >= 0.9)
    checks["ca_query_leaf_weight_le_0.2"] = bool(leaf_query <= 0.2)
    checks["ca_query_rmse_beats_insample"] = bool(rmse_q < rmse_in)
    # seed stability of the headline numbers
    checks["ca_leaf_gap_std_le_0.03"] = bool(ca_rows[2]["gap_std"] <= 0.03)
    checks["tw_leaf_gap_std_le_0.03"] = bool(report["ledger"]["taiwan"][2]["gap_std"] <= 0.03)

    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FLAG'}  {k}")
    report["criteria"] = checks
    report["all_pass"] = bool(all(checks.values()))

    (OUT / "m1_seeds.json").write_text(json.dumps(report, indent=2, default=float))
    # compact CSV for the paper tables
    pd.DataFrame(report["ledger"]["california"]).to_csv(OUT / "table1_california_ledger.csv", index=False)
    pd.DataFrame(report["ledger"]["taiwan"]).to_csv(OUT / "table2_taiwan_ledger.csv", index=False)
    print(f"\n{'ALL PASS' if report['all_pass'] else 'SOME FLAGS — inspect'} "
          f"— results in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
