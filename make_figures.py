"""Draw the paper's figures from results/*.json (run after utcf_sim.py)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#777777"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300})
W = 3.45
ld = lambda n: json.load(open(os.path.join(R, n + ".json")))

# Fig. 3 size of UTC
d = ld("size")
fig, ax = plt.subplots(figsize=(W, 2.4))
for reg, col, mk, lab in (("independent", ORANGE, "s", "Independent definitions"), ("harmonized", BLUE, "o", "Harmonized definitions")):
    js = sorted(int(k) for k in d[reg]); m = [d[reg][str(j)]["mean"] for j in js]
    lo = [d[reg][str(j)]["mean"] - d[reg][str(j)]["min"] for j in js]; hi = [d[reg][str(j)]["max"] - d[reg][str(j)]["mean"] for j in js]
    ax.errorbar(js, m, yerr=[lo, hi], color=col, marker=mk, ms=4, lw=1.5, capsize=2, label=lab)
ax.axhline(d["M"]["mean"], color=MUTED, ls="--", lw=0.8); ax.text(1, d["M"]["mean"] + 15, f"Bound M (distinct fact vectors) ≈ {d['M']['mean']:.0f}", fontsize=7, color=MUTED)
ax.axhline(d["patterns"]["mean"], color=MUTED, ls=":", lw=0.8); ax.text(20, d["patterns"]["mean"] - 75, f"Harmonized bound ≈ {d['patterns']['mean']:.0f}", fontsize=7, color=MUTED)
ax.set_xlabel("Number of jurisdictions J"); ax.set_ylabel("UTC classes K"); ax.set_xticks([1, 5, 10, 25, 50]); ax.set_ylim(0, 1000)
ax.legend(loc="center right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_utc_size.png")

# Fig. 4 change effort (log)
d = ld("change")
ev = ["E1", "E2", "E3", "E4", "E5"]
fig, ax = plt.subplots(figsize=(W, 2.4))
w = 0.26
for i, (k, col, lab, h) in enumerate((("B1", ORANGE, "Item codes, 3 systems", ""), ("B2", AQUA, "Central family code", "///"), ("UTCF", BLUE, "UTCF", ""))):
    xs = [x + (i - 1) * w for x in range(len(ev))]
    ys = [max(d[e][k]["median"], 0.5) for e in ev]
    ax.bar(xs, ys, w * 0.92, color=col, hatch=h, edgecolor="white", linewidth=0.5, label=lab)
    for x, e in zip(xs, ev):
        if d[e][k]["median"] == 0: ax.text(x, 0.6, "0", ha="center", fontsize=6, color=INK)
ax.set_yscale("log"); ax.set_ylim(0.5, 1e5); ax.set_xticks(range(len(ev))); ax.set_xticklabels(ev)
ax.set_ylabel("Records edited (median)"); ax.set_xlabel("Regulatory event"); ax.legend(fontsize=6.5, ncol=1, loc="upper right")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_change_effort.png")

# Fig. 5 error distribution (CCDF)
d = ld("errors")
fig, ax = plt.subplots(figsize=(W, 2.3))
for k, col, lab in (("B1", ORANGE, "Item codes, 3 systems"), ("UTCF", BLUE, "UTCF")):
    v = d[k]["volume_samples"]; n = len(v); xs = sorted(set(v))
    import bisect
    ys = [1 - bisect.bisect_left(v, x) / n for x in xs]
    ax.step(xs, ys, where="post", color=col, lw=1.5, label=lab)
ax.set_yscale("log"); ax.set_xlabel("Misclassified items per event (S-weighted)"); ax.set_ylabel("P(volume ≥ x)")
ax.set_xlim(0, 500); ax.set_ylim(1e-4, 1.2); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_error_ccdf.png")

# Fig. 6 runtime
d = ld("scale")
fig, ax = plt.subplots(figsize=(W, 2.3))
ns = sorted(int(k) for k in d["by_N"]); ax.plot(ns, [d["by_N"][str(n)]["seconds"] for n in ns], "o-", color=BLUE, lw=1.5, ms=4, label="Algorithm 1 (fact-vector signatures)")
nn = sorted(int(k) for k in d["naive_by_N"]); ax.plot(nn, [d["naive_by_N"][str(n)]["seconds"] for n in nn], "s--", color=ORANGE, lw=1.5, ms=4, label="Item-by-item classification")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Catalog size N (items)"); ax.set_ylabel("Runtime (s), J = 50"); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_runtime.png")
print("figures written")
