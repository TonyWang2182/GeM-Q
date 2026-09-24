"""Recompute study metrics from saved predictions, without rerunning inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
METHODS = ("erm", "gb", "sgb")
SEEDS = (42, 43, 44)


def load_predictions(root, method, seed, regime="ood"):
    names = (
        [f"ood{k}_{seed}_{method}" for k in range(5)]
        if regime == "ood"
        else [f"iid{seed}_{seed}_{method}"]
    )
    rows = []
    for name in names:
        folder = root / "results" / name
        if not (folder / "done.json").exists():
            raise FileNotFoundError(f"Incomplete run: {name}")
        rows.extend(json.loads((folder / "predictions.json").read_text()))
    rows.sort(key=lambda row: row["index"])
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Repeated held-out clip index")
    if regime == "ood" and len(rows) != 2748:
        raise ValueError("Expected exactly 2,748 outer-test predictions")
    return rows


def metrics(rows):
    targets = np.array([row["target"] for row in rows])
    predictions = np.array([row["prediction"] for row in rows])
    groups = np.array([row["system"] for row in rows])
    if not (np.isfinite(targets).all() and np.isfinite(predictions).all()):
        raise ValueError("Non-finite target or prediction")
    errors = abs(predictions - targets)
    group_mae = {g: float(errors[groups == g].mean()) for g in sorted(set(groups))}
    within = []
    for group in group_mae:
        selected = groups == group
        if (
            selected.sum() >= 5
            and np.ptp(targets[selected]) > 0
            and np.ptp(predictions[selected]) > 0
        ):
            within.append(
                float(spearmanr(predictions[selected], targets[selected]).statistic)
            )
    return {
        "mae": float(errors.mean()),
        "macro_mae": float(np.mean(list(group_mae.values()))),
        "q75_mae": float(np.quantile(list(group_mae.values()), 0.75)),
        "worst_mae": max(group_mae.values()),
        "srcc": float(spearmanr(predictions, targets).statistic),
        "pcc": float(pearsonr(predictions, targets).statistic),
        "within_srcc": float(np.mean(within)),
        "n_within": len(within),
        "group_mae": group_mae,
    }


def summarize(root, main_only=False):
    summary, seed_rows = {}, []
    for regime in ("iid", "ood"):
        summary[regime] = {}
        for method in METHODS:
            runs = [
                metrics(load_predictions(root, method, seed, regime)) for seed in SEEDS
            ]
            entry = {
                key: {
                    "mean": float(np.mean([run[key] for run in runs])),
                    "std": float(np.std([run[key] for run in runs], ddof=1)),
                }
                for key in runs[0]
                if key != "group_mae"
            }
            entry["group_mae"] = {
                group: float(np.mean([run["group_mae"][group] for run in runs]))
                for group in runs[0]["group_mae"]
            }
            summary[regime][method] = entry
            for seed, run in zip(SEEDS, runs):
                seed_rows.append(
                    [regime, method, seed]
                    + [
                        run[key]
                        for key in (
                            "mae",
                            "macro_mae",
                            "srcc",
                            "pcc",
                            "within_srcc",
                            "q75_mae",
                            "worst_mae",
                        )
                    ]
                )
    systems = sorted(summary["ood"]["erm"]["group_mae"])
    summary["paired"] = {}
    for method, baseline in (("gb", "erm"), ("sgb", "erm"), ("sgb", "gb")):
        differences = np.array(
            [
                summary["ood"][method]["group_mae"][g]
                - summary["ood"][baseline]["group_mae"][g]
                for g in systems
            ]
        )
        rng = np.random.default_rng(20260924)
        bootstrap = differences[
            rng.integers(0, len(systems), (5000, len(systems)))
        ].mean(1)
        summary["paired"][f"{method}-{baseline}"] = {
            "delta": float(differences.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "improved_generators": int((differences < 0).sum()),
            "total_generators": len(systems),
        }
    if not main_only:
        summary["sensitivity"] = {
            method: metrics(load_predictions(root, method, 42))
            for method in ("erm", "gb", "sgb", "dro", "lambda025", "eta005", "eta020")
        }
    return summary, seed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--main-only", action="store_true", help="Skip sensitivity runs"
    )
    args = parser.parse_args()
    summary, rows = summarize(args.root, args.main_only)
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "regime",
                "method",
                "seed",
                "MAE",
                "macro_MAE",
                "SRCC",
                "PCC",
                "within_SRCC",
                "q75_MAE",
                "worst_MAE",
            ]
        )
        writer.writerows(rows)
    for method in METHODS:
        result = summary["ood"][method]
        print(
            f"{method:>3}  macro MAE {result['macro_mae']['mean']:.4f}  pooled SRCC {result['srcc']['mean']:.4f}"
        )
    print(f"Numerical summaries written to {output}")


if __name__ == "__main__":
    main()
