"""
Fits a power-law tail to the in-degree distribution using the
maximum-likelihood estimator of Clauset, Shalizi & Newman (2009).

Also fits a log-normal for comparison.
Saves degree_distribution.png (with fit lines) to results/figures/.

"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import kstest

state = pickle.load(open("results/.cache/build_state.pkl", "rb"))
g_dir = state["g_dir"]
indeg = np.array(g_dir.indegree())


# Scan k_min candidates and pick the one minimising the KS statistic
def mle_alpha(data, k_min):
    tail = data[data >= k_min]
    n    = len(tail)
    if n < 10:
        return None
    alpha = 1 + n / np.sum(np.log(tail / (k_min - 0.5)))
    return alpha

def ks_stat(data, k_min, alpha):
    tail  = data[data >= k_min]
    n     = len(tail)
    ks    = np.sort(tail)
    # empirical CDF
    emp   = np.arange(1, n + 1) / n
    # theoretical CDF: P(K <= k) = 1 - (k/k_min)^(1-alpha)
    theo  = 1 - (ks / k_min) ** (1 - alpha)
    return np.max(np.abs(emp - theo))

candidates = np.arange(2, 200)
results    = []
for k in candidates:
    a = mle_alpha(indeg, k)
    if a is not None and a > 1:
        ks = ks_stat(indeg, k, a)
        results.append((ks, k, a))

results.sort()
best_ks, k_min, alpha = results[0]

print(f"Power-law fit:")
print(f"  k_min = {k_min}")
print(f"  alpha = {alpha:.4f}")
print(f"  KS    = {best_ks:.4f}")
print(f"  Tail  = {(indeg >= k_min).sum():,} nodes ({100*(indeg>=k_min).mean():.1f}% of all nodes)")

# ── Log-normal fit ─────────────────────────────────────────────────────────────
from scipy.stats import lognorm
pos = indeg[indeg > 0]
mu_ln, sigma_ln = np.log(pos).mean(), np.log(pos).std()
print(f"\nLog-normal fit (k > 0):")
print(f"  mu_log = {mu_ln:.3f}, sigma_log = {sigma_ln:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
Path("results/figures").mkdir(parents=True, exist_ok=True)

counts  = np.bincount(indeg)
ks_vals = np.where(counts > 0)[0]
ps_vals = counts[ks_vals] / counts[ks_vals].sum()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(ks_vals, ps_vals, s=5, alpha=0.5, color="#4e79a7", label="Empirical", zorder=3)

# power-law line
k_range = np.linspace(k_min, ks_vals.max(), 300)
C = (alpha - 1) / k_min                    # normalisation constant
pl_y = C * (k_range / k_min) ** (-alpha)
ax.plot(k_range, pl_y / pl_y[0] * ps_vals[ks_vals >= k_min][0],
        color="#e15759", linewidth=1.5,
        label=rf"Power law $\alpha={alpha:.2f}$, $k_{{\min}}={k_min}$")

# log-normal line
from scipy.stats import lognorm as ln_dist
x_ln = np.linspace(1, ks_vals.max(), 300)
pdf  = ln_dist.pdf(x_ln, s=sigma_ln, scale=np.exp(mu_ln))
ax.plot(x_ln, pdf / pdf.max() * ps_vals.max(),
        color="#59a14f", linewidth=1.5, linestyle="--", label="Log-normal")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("In-degree $k$"); ax.set_ylabel("$P(k)$")
ax.set_title("In-degree Distribution with Fits (log–log)")
ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.3)
sns.despine()
fig.tight_layout()
fig.savefig("results/figures/degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved updated degree_distribution.png with fit lines.")

