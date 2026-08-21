import os
import matplotlib.pyplot as plt
import pandas as pd

plot_mean = True
csv = 'multi_seed_result/csv/baseline_compare.csv'


ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    df = pd.read_csv(csv)
    num_cols = [c for c in df.columns if c != "label"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    stats = [{
        "label": row["label"],
        "med": row["mid"],
        "q1": row["q1"],
        "q3": row["q3"],
        "whislo": row["min"],
        "whishi": row["max"],
        "mean": row["mean"],
    } for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    box = ax.bxp(stats, showfliers=False, showmeans=plot_mean, meanline=plot_mean)
    ax.set_xlabel("label")
    ax.set_ylabel("test_dev")
    ax.grid(alpha=0.1, axis="y")
    if plot_mean:
        ax.legend([box["medians"][0], box["means"][0]], ["median", "mean"])
    fig.tight_layout()

    out = os.path.join(ROOT, "box.png")
    fig.savefig(out, dpi=150)
    print(f"已存 {out}")


main()
