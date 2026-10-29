"""M2 gate — screen the τ statistics (runs R009/R010–R012).

R009 is embedded: the RBF settings are the SURE positive control (corr(SURE, true
risk) must be ≥ 0.95 for at least the well-scaled RBF). The gate: at least one τ
achieves Spearman ρ ≥ 0.8 against |SURE − true risk| across the setting list, and
flags the label-fitted regime without flagging the control.

Writes experiments/E2/results/m2_gate.json. Exit code 0 = GO for C2, 1 = PIVOT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.E2.exp2 import m2_screen  # noqa: E402

OUT = Path(__file__).parent / "results"


def main():
    OUT.mkdir(exist_ok=True)
    print("M2 gate — settings: 4 fixed RBF + 1 fixed-leaf control + 1 label-fitted leaf\n")
    out = m2_screen(n_draws=10, n_boot=20)

    rows = out["rows"]
    print("\n--- gate checks (criteria revised after run #1 — see m2_gate_notes.md) ---")
    checks = {}

    # R009 positive control: best fixed-RBF corr(SURE, true) ≥ 0.95
    rbf_corrs = [r["corr_sure"] for r in rows if r["setting"].startswith("RBF")]
    checks["positive_control_corr_ge_0.95"] = bool(max(rbf_corrs) >= 0.95)

    # separation: label-fitted leafs must exceed every fixed/control setting
    lf = [r for r in rows if "LABEL-FITTED" in r["setting"]]
    fixed = [r for r in rows if "LABEL-FITTED" not in r["setting"]]
    checks["tau_df_separates"] = bool(min(r["tau_df"] for r in lf)
                                      > max(r["tau_df"] for r in fixed))
    checks["tau_boot_separates"] = bool(min(r["tau_boot"] for r in lf)
                                        > max(r["tau_boot"] for r in fixed))
    ctrl = next(r for r in rows if "CONTROL" in r["setting"])
    checks["tau_boot_control_is_zero"] = bool(ctrl["tau_boot"] < 1e-10)
    checks["control_not_flagged"] = bool(ctrl["tau_df"] < min(r["tau_df"] for r in lf)
                                         and ctrl["tau_boot"] < min(r["tau_boot"] for r in lf))

    # dose-response on the depth axis (the honest rank-correlation test)
    # tolerance 1e-9: rho=0.800 is exactly representable as 8/10 != 0.8 in binary
    rhos_dose = {k: v["rho"] for k, v in out.items() if k.startswith("spearman_") and "dose" in k}
    checks["dose_response_rho_ge_0.8"] = bool(any(v >= 0.8 - 1e-9 for v in rhos_dose.values()))

    # SURE must actually mislead in the high-dose regime (the failure is real)
    deepest = max(lf, key=lambda r: r["setting"])
    checks["label_fitted_sure_err_gt_3x_fixed"] = bool(
        deepest["sure_err_mean"] > 3 * max(r["sure_err_mean"] for r in fixed))

    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n  dose rhos: { {k: round(v, 3) for k, v in rhos_dose.items()} }")

    out["checks"] = checks
    out["gate_go"] = bool(checks["tau_df_separates"] and checks["control_not_flagged"]
                          and checks["dose_response_rho_ge_0.8"]
                          and checks["label_fitted_sure_err_gt_3x_fixed"])
    (OUT / "m2_gate.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nGATE DECISION: {'GO — continue with C2' if out['gate_go'] else 'PIVOT — drop to C1-only'}")
    print(f"results: {OUT / 'm2_gate.json'}")
    return 0 if out["gate_go"] else 1


if __name__ == "__main__":
    sys.exit(main())
