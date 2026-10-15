"""Builds the paper's figures from the notebook's results (training logs, prediction files, result tables).

usage: python make_figures.py <results folder> <folder with common.py (src/ in the public release)> <output folder>
Colors follow the entity across the paper (validated with the dataviz palette validator, light mode):
R1 blue, R3 orange, R2 aqua, R1b yellow, second base model magenta, AdaptThink green, Thinkless red,
R1-Distill violet; base model and SFT are neutral grey references. Line style is a second cue everywhere.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT, BUILD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(BUILD))
import common  # noqa: E402  (the notebook's own answer normalisation for votes)

OUT.mkdir(parents=True, exist_ok=True)
C = dict(R1="#2a78d6", R3="#eb6834", R2="#1baf7a", R1b="#eda100", M2="#e87ba4", ADAPT="#008300", THINKLESS="#e34948",
         DISTILL="#6250d6", BASE="#7a7a7a", SFT="#3d3d3a", BATCH="#2a78d6", NONE="#1baf7a", GROUP="#e34948")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round", "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42})
DOUBLE, SINGLE = 7.16, 3.5          # IEEE column widths (inches)


def read_jsonl(p):
    """Reads <name>.jsonl, or <name>.jsonl.gz as in the public release."""
    if not p.exists():
        import gzip
        return [json.loads(l) for l in gzip.open(str(p) + ".gz", "rt", encoding="utf-8") if l.strip()]
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def log(run):
    h = pd.DataFrame(json.loads((ROOT / "runs" / run / "trainer_log_history.json").read_text()))
    return h.dropna(subset=["rewards/strict_accuracy/mean"]).set_index("step").sort_index()


def smooth(s, w=5):
    return s.rolling(w, min_periods=1, center=True).mean()


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure: GRPO learns not to think
fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3))
for run, lab, col, ls in (("grpo_ours", "R1 (3 seeds)", C["R1"], "-"), ("grpo_stdlen", "R3", C["R3"], "--"),
                          ("grpo_judge", "R2", C["R2"], "-."), ("m2_grpo", "R1, second base model", C["M2"], ":")):
    h = log(run)
    a.plot(h.index, 100 * smooth(h["rewards/think_rate/mean"]), color=col, ls=ls, label=lab)
    if run == "grpo_ours":
        for s in ("grpo_ours_s2", "grpo_ours_s3"):
            hs = log(s)
            a.plot(hs.index, 100 * smooth(hs["rewards/think_rate/mean"]), color=col, lw=0.7, alpha=0.6)
hb = log("grpo_balanced")
a.plot(hb.index, 100 * smooth(hb["rewards/policy_think_prob/mean"]), color=C["R1b"], ls=(0, (5, 1, 1, 1)),
       label="R1b, policy's own P(think)")
a.set_xlabel("GRPO step"); a.set_ylabel("think-rate of rollouts (%)"); a.set_ylim(-2, 62); a.set_xlim(0, 200)
a.set_title("(a) Choice of the think mode during training", loc="left")
a.legend(loc="upper right")
for run, lab, col in (("grpo_ours", "Qwen2.5-Math-1.5B-Instruct (R1)", C["R1"]), ("m2_grpo", "Qwen2.5-1.5B-Instruct (R1)", C["M2"])):
    h = log(run).loc[1:60]
    tr, tc, sa = h["rewards/think_rate/mean"], h["rewards/think_correct/mean"], h["rewards/strict_accuracy/mean"]
    w = 5                                       # trailing 5-step window; n = steps actually in the window
    n = sa.rolling(w, min_periods=1).count()
    think_share = tr.rolling(w, min_periods=1).sum()
    think_acc = (tc.rolling(w, min_periods=1).sum() / think_share).where(think_share / n >= 0.05)
    short_acc = (sa.rolling(w, min_periods=1).sum() - tc.rolling(w, min_periods=1).sum()) / (n - think_share)
    b.plot(h.index, 100 * short_acc, color=col, ls="-", label=f"{lab}: short")
    b.plot(h.index, 100 * think_acc, color=col, ls="--", label=f"{lab}: think")
b.set_xlabel("GRPO step"); b.set_ylabel("rollout accuracy by mode (%)"); b.set_xlim(0, 60); b.set_ylim(0, 100)
b.set_title("(b) Accuracy of think and short rollouts", loc="left")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=1.0)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_mode_dynamics")

# ---------------------------------------------------------------- Figure: think minus short by MATH level
t16 = pd.read_csv(ROOT / "paper_tables" / "table16_think_vs_short_by_level.csv")
t19c = pd.read_csv(ROOT / "paper_tables" / "table19c_second_model_think_vs_short.csv")
fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
series = [(t16, "sft_v2", "SFT", C["SFT"], "-", "o", True), (t16, "grpo_balanced", "R1b", C["R1b"], "-", "s", True),
          (t16, "grpo_ours", "R1", C["R1"], "-", "D", True),
          (t19c, "m2_sft", "SFT, second base", C["SFT"], "--", "o", False),
          (t19c, "m2_grpo", "R1, second base", C["R1"], "--", "D", False)]
for j, (t, key, lab, col, ls, mk, filled) in enumerate(series):
    x = t[(t.key == key) & (t.subset != "all")]
    if "bench" in x:
        x = x[x.bench == "math500"]
    lv = x.subset.str[1:].astype(int).values + (j - 2) * 0.08
    y, lo, hi = 100 * x.think_minus_short.values, 100 * x.ci_low.values, 100 * x.ci_high.values
    ax.errorbar(lv, y, yerr=[y - lo, hi - y], color=col, ls=ls, marker=mk, ms=4, lw=1.1, capsize=1.5, elinewidth=0.7,
                mfc=col if filled else "white", mec=col, label=lab)
ax.axhline(0, color="#52514e", lw=0.7)
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([f"L{i}" for i in range(1, 6)])
ax.set_xlabel("MATH-500 difficulty level"); ax.set_ylabel("forced think $-$ forced short (points)")
h, l = ax.get_legend_handles_labels()        # columns: SFT pair, R1 pair, R1b
order = [l.index(x) for x in ("SFT", "SFT, second base", "R1", "R1, second base", "R1b")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          columnspacing=0.8, handlelength=2.2)
fig.tight_layout()
save(fig, "fig_think_short_levels")

# ---------------------------------------------------------------- Figure: length collapse under group-std advantages
fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
for run, lab, col, ls in (("grpo_ours", "R1: batch std, bonus", C["BATCH"], "-"),
                          ("grpo_ours_nostd", "C2: no std scaling, bonus", C["NONE"], "-"),
                          ("grpo_nobonus_gstd", "C1: group std, no bonus", C["GROUP"], "--"),
                          ("grpo_ours_gstd", "R1 with group std, bonus", C["GROUP"], "-"),
                          ("grpo_ours_gstd_b01", "C3: group std, bonus $\\times$0.1", C["GROUP"], ":")):
    h = log(run)
    a.plot(h.index, smooth(h["completions/mean_length"], 9), color=col, ls=ls, label=lab)
a.set_title("(a) Qwen2.5-Math-1.5B-Instruct", loc="left")
for run, lab, col, ls in (("m2_grpo", "R1: batch std, bonus", C["BATCH"], "-"),
                          ("m2_grpo_gstd", "R1 with group std, bonus", C["GROUP"], "-")):
    h = log(run)
    b.plot(h.index, smooth(h["completions/mean_length"], 9), color=col, ls=ls, label=lab)
b.set_title("(b) Qwen2.5-1.5B-Instruct", loc="left")
for ax in (a, b):
    ax.set_xlabel("GRPO step"); ax.set_ylabel("mean completion length (tokens)"); ax.set_xlim(0, 200)
    ax.axvspan(61, 100, color="#f0efec", lw=0, zorder=0); ax.axvspan(161, 200, color="#f0efec", lw=0, zorder=0)
    top = ax.get_ylim()[1]
    for x0, lab in ((80.5, "mid"), (180.5, "late")):
        ax.text(x0, top, lab, ha="center", va="top", fontsize=6.5, color="#52514e")
h, l = a.get_legend_handles_labels()          # panel (b) reuses two of these five styles
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), columnspacing=1.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_collapse")

# ---------------------------------------------------------------- Figure: breadth vs depth at matched tokens
def akey(a):
    r = common.rational(a)
    return str(r) if r is not None else str(a).strip()


def vote_credit(answers, corrects):
    votes = Counter("<correct>" if c else akey(a) for a, c in zip(answers, corrects) if c or (isinstance(a, str) and a.strip()))
    if not votes:
        return 0.0
    top = max(votes.values())
    tied = [a for a, v in votes.items() if v == top]
    return ("<correct>" in tied) / len(tied)


t18a = pd.read_csv(ROOT / "paper_tables" / "table18a_multiple_attempts.csv")
t18b = pd.read_csv(ROOT / "paper_tables" / "table18b_breadth_vs_depth.csv")
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
check = {}
for ax, bench, title in zip(axes, ("gsm8k", "math500"), ("(a) GSM8K", "(b) MATH-500")):
    for key, lab, col, ls in (("base", "Base model, vote of $k$", C["BASE"], "--"), ("grpo_ours", "R1, vote of $k$", C["R1"], "-")):
        v = pd.DataFrame(read_jsonl(ROOT / "eval" / "preds" / f"vote8__{key}.jsonl"))
        v = v[v.bench == bench].sort_values(["pid", "sample"])
        g = {pid: (list(x.final_answer), list(x.correct.astype(bool)), list(x.n_tokens)) for pid, x in v.groupby("pid")}
        acc = [np.mean([vote_credit(a[:k], c[:k]) for a, c, _ in g.values()]) for k in range(1, 9)]
        tok = [np.mean([sum(t[:k]) for _, _, t in g.values()]) for k in range(1, 9)]
        check[(key, bench)] = acc[-1]
        ax.plot(tok, [100 * x for x in acc], color=col, ls=ls, marker="o", ms=2.5, label=lab)
    for dkey, lab, col, mk in (("grpo_ours", "R1, forced think", C["R1"], "^"), ("adaptthink", "AdaptThink-1.5B", C["ADAPT"], "s"),
                               ("thinkless", "Thinkless-1.5B", C["THINKLESS"], "D"), ("r1distill", "R1-Distill-1.5B", C["DISTILL"], "v")):
        r = t18b[(t18b.bench == bench) & (t18b.depth_key == dkey) & t18b.vote_source.str.startswith("full")].iloc[0]
        ax.plot(r.depth_tokens, 100 * r.depth_acc, ls="none", marker=mk, ms=6, mfc=col, mec="white", mew=0.8,
                label=f"{lab} (one answer)")
    ax.set_xscale("log"); ax.set_xlabel("mean generated tokens per problem (log scale)")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([300, 500, 1000, 2000, 4000]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("accuracy (%)"); ax.set_title(title, loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(w_pad=2.0)
save(fig, "fig_breadth_depth")
want = t18a[t18a.vote_source.str.startswith("full")].set_index(["model", "bench"]).majority_at_K
for (key, bench), acc in check.items():     # the recomputed maj@8 must equal the notebook's Table 18a
    lab = "Base (Qwen2.5-Math-1.5B-Inst)" if key == "base" else "GRPO ours (R1)"
    assert abs(acc - want.loc[(lab, bench)]) < 1e-9, (key, bench, acc, want.loc[(lab, bench)])
print("vote curves match Table 18a")

# ---------------------------------------------------------------- Figure: accuracy under token budgets
t15 = pd.read_csv(ROOT / "paper_tables" / "table15_budget_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4))
for ax, bench, title in zip(axes, ("gsm8k", "math500"), ("(a) GSM8K", "(b) MATH-500")):
    for key, lab, col, ls, mk in (("grpo_ours", "R1 (ours)", C["R1"], "-", "o"), ("base", "Base model", C["BASE"], "--", "o"),
                                  ("adaptthink", "AdaptThink-1.5B", C["ADAPT"], "-", "s"),
                                  ("thinkless", "Thinkless-1.5B", C["THINKLESS"], "-.", "D"),
                                  ("r1distill", "R1-Distill-1.5B", C["DISTILL"], ":", "v")):
        x = t15[(t15.model == key) & (t15.bench == bench)].sort_values("budget")
        ax.plot(x.budget, 100 * x.acc, color=col, ls=ls, marker=mk, ms=3, label=lab)
        if key in ("grpo_ours", "adaptthink"):
            ax.fill_between(x.budget, 100 * x.ci_low, 100 * x.ci_high, color=col, alpha=0.12, lw=0)
    ax.set_xscale("log", base=2); ax.set_xticks([256, 512, 1024, 2048, 3584, 8192])
    ax.set_xticklabels(["256", "512", "1024", "2048", "3584", "8192"])
    ax.set_xlabel("token budget (maximum generated tokens)"); ax.set_ylabel("accuracy (%)"); ax.set_title(title, loc="left")
axes[1].legend(loc="lower right")
fig.tight_layout(w_pad=2.0)
save(fig, "fig_budget")
