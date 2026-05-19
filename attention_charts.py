import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

views = [f"Image {i+1}" for i in range(12)]
x = np.arange(12)

# Chart 1: Equal weight
equal = [8.33] * 12
ax1.bar(x, equal, color="#3b82f6", edgecolor="white", linewidth=0.5)
ax1.set_title("Without Attention\nEqual Weight", fontsize=14, fontweight="bold", color="white")
ax1.set_ylim(0, 30)
ax1.set_xticks(x)
ax1.set_xticklabels(views, rotation=45, ha="right", fontsize=8, color="white")
ax1.set_ylabel("Weight %", color="white", fontsize=12)
ax1.tick_params(colors="white")
ax1.set_facecolor("#1a1a2e")
for spine in ax1.spines.values():
    spine.set_color("#444")

# Chart 2: Learned attention
attention = [25, 22, 15, 12, 8, 5, 4, 3, 2, 2, 1, 1]
colors = ["#22c55e" if v >= 10 else "#e94560" if v <= 2 else "#eab308" for v in attention]
ax2.bar(x, attention, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_title("With Attention\nLearned Weight", fontsize=14, fontweight="bold", color="white")
ax2.set_ylim(0, 30)
ax2.set_xticks(x)
ax2.set_xticklabels(views, rotation=45, ha="right", fontsize=8, color="white")
ax2.set_ylabel("Weight %", color="white", fontsize=12)
ax2.tick_params(colors="white")
ax2.set_facecolor("#1a1a2e")
for spine in ax2.spines.values():
    spine.set_color("#444")

fig.patch.set_facecolor("#1a1a2e")
plt.tight_layout()
plt.savefig("/Users/danarfi/Desktop/2d3d/submission/v3/attention_charts.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print("Saved to attention_charts.png")
