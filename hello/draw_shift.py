import csv
import os

import matplotlib.pyplot as plt
import numpy as np

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_result.csv")

data = {}
amounts = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        cat = row["category"]
        amount = int(row["amount"])
        offset = float(row["avg_offset"])
        data.setdefault(cat, {})[amount] = offset
        if amount not in amounts:
            amounts.append(amount)

categories = list(data.keys())
x = np.arange(len(categories))
width = 0.8 / len(amounts)

fig, ax = plt.subplots(figsize=(12, 6))
for i, amount in enumerate(amounts):
    values = [data[cat][amount] for cat in categories]
    ax.bar(x + i * width, values, width, label=f"+{amount}")

ax.set_xticks(x + width * (len(amounts) - 1) / 2)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.set_ylabel("avg offset")
ax.set_title("Latent Shift by Category")
ax.legend(title="amount")
fig.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shift_result.png")
fig.savefig(out_path)
print(f"saved to {out_path}")
