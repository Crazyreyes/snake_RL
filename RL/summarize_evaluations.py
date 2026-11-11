import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

EVALUATION_DIR = Path(__file__).resolve().parent / "evaluations"

AGENT_ORDER = ["standard", "improved", "improved_double"]
LOSS_ORDER = ["huber", "mse", "unknown"]


def load_evaluations(directory):
    """Return one entry per (loss, agent, seed), latest file wins."""
    entries = {}
    for path in sorted(directory.glob("snake_*_evaluation_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metrics = data.get("metrics") or {}
        if "average_score" not in metrics:
            continue
        key = (
            str(data.get("loss", "unknown")),
            str(data.get("experiment", "unknown")),
            str(data.get("seed", "unknown")),
        )
        # Files are timestamped; the later file for the same run wins.
        entries[key] = {
            "loss": key[0],
            "agent": key[1],
            "seed": key[2],
            "timestamp": data.get("evaluation_timestamp", ""),
            "path": path.name,
            "metrics": metrics,
        }
    return list(entries.values())


def sort_key(entry):
    loss = LOSS_ORDER.index(entry["loss"]) if entry["loss"] in LOSS_ORDER else 99
    agent = (
        AGENT_ORDER.index(entry["agent"])
        if entry["agent"] in AGENT_ORDER
        else 99
    )
    return (loss, agent, str(entry["seed"]))


def per_run_rows(entries):
    rows = []
    for e in sorted(entries, key=sort_key):
        m = e["metrics"]
        rows.append(
            [
                e["loss"],
                e["seed"],
                e["agent"],
                f"{m['average_score']:.2f}",
                f"{m['median_score']}",
                f"{m['score_std_dev']:.2f}",
                f"{m['best_score']}",
                f"{m['worst_score']}",
                f"{m['average_steps']:.2f}",
                f"{m['truncated_games']}",
            ]
        )
    return rows


def across_seed_rows(entries):
    """Mean across seeds for each (loss, agent), plus Huber - MSE."""
    grouped = defaultdict(list)
    for e in entries:
        grouped[(e["loss"], e["agent"])].append(e)

    rows = []
    for (loss, agent) in sorted(
        grouped,
        key=lambda k: (
            LOSS_ORDER.index(k[0]) if k[0] in LOSS_ORDER else 99,
            AGENT_ORDER.index(k[1]) if k[1] in AGENT_ORDER else 99,
        ),
    ):
        group = grouped[(loss, agent)]
        avgs = [g["metrics"]["average_score"] for g in group]
        steps = [g["metrics"]["average_steps"] for g in group]
        rows.append(
            [
                loss,
                agent,
                str(len(group)),
                f"{mean(avgs):.2f}",
                f"{pstdev(avgs):.2f}" if len(avgs) > 1 else "0.00",
                f"{min(avgs):.2f}",
                f"{max(avgs):.2f}",
                f"{mean(steps):.2f}",
                f"{sum(g['metrics']['truncated_games'] for g in group)}",
            ]
        )

    # Huber minus MSE, per agent, where both exist.
    for agent in AGENT_ORDER:
        huber = [e for e in entries if e["loss"] == "huber" and e["agent"] == agent]
        mse = [e for e in entries if e["loss"] == "mse" and e["agent"] == agent]
        if huber and mse:
            diff = mean([e["metrics"]["average_score"] for e in huber]) - mean(
                [e["metrics"]["average_score"] for e in mse]
            )
            rows.append(
                [
                    "huber - mse",
                    agent,
                    f"{len(huber)} vs {len(mse)}",
                    f"{diff:+.2f}",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return rows


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print(" | ".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=str(EVALUATION_DIR),
        help="Directory of evaluation JSONs (default: RL/evaluations).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Also write the tables to this .md file (and a .csv beside it).",
    )
    args = parser.parse_args()

    directory = Path(args.dir)
    entries = load_evaluations(directory)
    if not entries:
        print(f"No evaluation JSONs found in {directory}")
        return 1

    run_headers = [
        "Loss",
        "Seed",
        "Agent",
        "Eval avg",
        "Median",
        "Std dev",
        "Best",
        "Worst",
        "Avg steps",
        "Truncated",
    ]
    agg_headers = [
        "Loss",
        "Agent",
        "Seeds",
        "Mean eval avg",
        "Std across seeds",
        "Min",
        "Max",
        "Mean steps",
        "Truncated",
    ]

    run_rows = per_run_rows(entries)
    agg_rows = across_seed_rows(entries)

    print(f"\nEvaluations: {len(entries)} runs from {directory}\n")
    print("Per run")
    print_table(run_headers, run_rows)
    print("\nAveraged across seeds")
    print_table(agg_headers, agg_rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as file:
            file.write("# Evaluation summary\n\n")
            file.write("## Per run\n\n")
            file.write(" | ".join(run_headers) + "\n")
            file.write(" | ".join("---" for _ in run_headers) + "\n")
            for row in run_rows:
                file.write(" | ".join(row) + "\n")
            file.write("\n## Averaged across seeds\n\n")
            file.write(" | ".join(agg_headers) + "\n")
            file.write(" | ".join("---" for _ in agg_headers) + "\n")
            for row in agg_rows:
                file.write(" | ".join(row) + "\n")
        csv_path = out.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(run_headers)
            writer.writerows(run_rows)
        print(f"\nWrote {out} and {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())