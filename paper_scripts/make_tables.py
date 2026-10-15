"""Builds the paper's LaTeX tables from the notebook's result files (paper_tables/*.csv).

usage: python make_tables.py <results folder: .../slmreasoning> <output folder: paper/tables>
Every number comes from the CSVs; nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

import pandas as pd

SRC, OUT = Path(sys.argv[1]) / "paper_tables", Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
T = lambda name: pd.read_csv(SRC / f"{name}.csv")


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.1f}"


def cell(acc, lo, hi):
    return f"{pct(acc)} {{\\scriptsize[{pct(lo)}, {pct(hi)}]}}"


def delta(x):
    return "--" if pd.isna(x) else f"${100 * x:+.1f}$"


def pv(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def bold_if(s, cond):
    """Bold a cell; numbers in math mode need \\boldmath (\\textbf does not reach math)."""
    if not cond:
        return s
    return f"{{\\boldmath {s}}}" if s.lstrip().startswith("$") else f"\\textbf{{{s}}}"


def write(name, text):
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", name)


NAME = {
    "base": "Qwen2.5-Math-1.5B-Instruct (base)",
    "r1distill": "DeepSeek-R1-Distill-Qwen-1.5B",
    "thinkless": "Thinkless-1.5B",
    "adaptthink": "AdaptThink-1.5B",
    "sft_long": "SFT, long traces ($\\le$3{,}400 tokens)",
    "sft_v2": "SFT",
    "dpo_v2": "DPO",
    "grpo_ours": "R1: GRPO + difficulty-aware reward",
    "grpo_judge": "R2: R1 with LLM-judge reward",
    "grpo_stdlen": "R3: GRPO + standard length bonus",
    "grpo_balanced": "R1b: R1 + balanced mode exploration$^{a}$",
    "grpo_ours_gstd": "R1 with group-std advantages",
    "grpo_stdlen_gstd": "R3 with group-std advantages",
}

# ---------------------------------------------------------------- Table: main results
t1 = T("table1_main_results")
s1 = t1[t1.condition == "single"].set_index(["model", "bench"])
rows = []
groups = [("Untrained models", ["base", "r1distill", "thinkless", "adaptthink"]),
          ("Trained from the base model (ours)", ["sft_long", "sft_v2", "dpo_v2", "grpo_ours", "grpo_judge",
                                                 "grpo_stdlen", "grpo_balanced", "grpo_ours_gstd", "grpo_stdlen_gstd"])]
for title, keys in groups:
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}}\\\\")
    for k in keys:
        r = {b: s1.loc[(k, b)] for b in ("gsm8k", "math500", "aime2026")}
        rows.append(" & ".join([NAME[k]] + [cell(r[b].acc, r[b].ci_low, r[b].ci_high) for b in ("gsm8k", "math500", "aime2026")]
                               + [f"{r['math500'].mean_tokens:,.0f}".replace(",", "{,}"),
                                  "--" if k == "base" else pct(r["math500"].think_rate)]) + "\\\\")   # base: no modes
        if k == "grpo_ours":
            sd = T("table14_seed_robustness").set_index("bench")
            rows.append(" & ".join(["\\quad mean $\\pm$ sd over 3 seeds"] +
                                   [f"{100 * sd.loc[b, 'mean']:.1f} $\\pm$ {100 * sd.loc[b, 'sd']:.1f}" for b in ("gsm8k", "math500", "aime2026")]
                                   + ["--", "--"]) + "\\\\")
    rows.append("\\midrule")
rows = rows[:-1]
write("tab_main", r"""\begin{table*}[!t]
\centering
\caption{Strict accuracy (\%) with Wilson 95\% intervals, greedy decoding, on GSM8K ($n=1{,}319$), MATH-500 ($n=500$)
and AIME 2026 ($n=30$), with the mean number of generated tokens and the think-rate on MATH-500. The dual-mode models
choose their own mode. R1-Distill, Thinkless and AdaptThink may generate up to 8{,}192 tokens; all other models up to
3{,}584.}
\label{tab:main}
\footnotesize
\begin{tabular}{@{}lccccc@{}}
\toprule
Model & GSM8K & MATH-500 & AIME 2026 & Tokens (MATH) & Think-rate (MATH) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}\raggedright\footnotesize $^{a}$R1b uses group-std advantages (it was trained before the switch to batch
std and did not collapse).
\end{table*}
""")

# ---------------------------------------------------------------- Table: think vs short (forced modes, same problems)
t16 = T("table16_think_vs_short_by_level")
t19c = T("table19c_second_model_think_vs_short")
t19b = T("table19b_second_model_paired").set_index(["comparison", "bench"])
rows = []
spec = [("Qwen2.5-Math-1.5B-Instruct", [("sft_v2", "SFT"), ("grpo_balanced", "R1b"), ("grpo_ours", "R1")]),
        ("Qwen2.5-1.5B-Instruct", [("m2_sft", "SFT"), ("m2_grpo", "R1")])]
for base, models in spec:
    rows.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{base}}}}}\\\\")
    for key, lab in models:
        if key.startswith("m2_"):
            d = t19c[t19c.key == key].set_index("subset")
            comp = "SFT: forced think − forced short" if key == "m2_sft" else "GRPO R1: forced think − forced short"
            g = t19b.loc[(comp, "gsm8k")]
            gs = bold_if(delta(g.delta), g.p_holm < 0.05)
        else:
            d = t16[(t16.key == key) & (t16.bench == "math500")].set_index("subset")
            g = t16[(t16.key == key) & (t16.bench == "gsm8k")].iloc[0]
            gs = bold_if(delta(g.think_minus_short), g.p_holm < 0.05)
        a = d.loc["all"]
        cells = [lab, gs, f"{pct(a.acc_think)} / {pct(a.acc_short)}", bold_if(delta(a.think_minus_short), a.p_holm < 0.05)]
        cells += [bold_if(delta(d.loc[f"L{i}", "think_minus_short"]), d.loc[f"L{i}", "p_holm"] < 0.05) for i in range(1, 6)]
        rows.append(" & ".join(cells) + "\\\\")
    rows.append("\\midrule")
rows = rows[:-1]
write("tab_think_short", r"""\begin{table*}[!t]
\centering
\caption{Does thinking pay off? Forced think minus forced short accuracy (percentage points) on the same problems.
MATH-500 shows the two accuracies and the difference overall and by difficulty level (L1 easiest). Bold: significant
after Holm correction (exact McNemar; Holm within each source table).}
\label{tab:thinkshort}
\footnotesize
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & GSM8K & \multicolumn{2}{c}{MATH-500} & \multicolumn{5}{c}{MATH-500 by level} \\
\cmidrule(lr){3-4}\cmidrule(l){5-9}
Model & $\Delta$ & think / short & $\Delta$ & L1 & L2 & L3 & L4 & L5 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: length collapse (mechanism)
t17 = T("table17_mechanism_controls").set_index("run")
t19d = T("table19d_second_model_dynamics").set_index("run")
t19a = T("table19a_second_model")
m2_math = {k: t19a[(t19a.key == k) & (t19a.condition == "single") & (t19a.bench == "math500")].acc.iloc[0]
           for k in ("m2_grpo", "m2_grpo_gstd")}
spec = [("grpo_ours", "R1", "batch", "DA"), ("grpo_ours_s2", "R1, seed 2", "batch", "DA"),
        ("grpo_ours_s3", "R1, seed 3", "batch", "DA"), ("grpo_judge", "R2 (judge reward)", "batch", "DA"),
        ("grpo_stdlen", "R3", "batch", "length"), ("grpo_ours_nostd", "C2", "none", "DA"),
        ("grpo_nobonus_gstd", "C1", "group", "none"), ("grpo_balanced", "R1b (balanced)", "group", "DA"),
        ("grpo_ours_gstd", "R1, group std", "group", "DA"), ("grpo_stdlen_gstd", "R3, group std", "group", "length"),
        ("grpo_ours_gstd_b01", "C3", "group", "DA $\\times$0.1")]
rows = ["\\multicolumn{9}{l}{\\textit{Qwen2.5-Math-1.5B-Instruct}}\\\\"]


def mech_row(lab, sc, bonus, r, greedy):
    acg = r.get("all_correct_groups_mid", float("nan"))
    col = "yes" if bool(r["collapsed"]) else "no"
    return " & ".join([lab, sc, bonus, f"{r['length_mid']:.0f}", f"{r['length_late']:.0f}",
                       f"{r['length_ratio_late_mid']:.2f}", bold_if(col, col == "yes"), pct(r["acc_late"]),
                       pct(acg), pct(greedy)]) + "\\\\"


for run, lab, sc, bonus in spec:
    r = t17.loc[run]
    rows.append(mech_row(lab, sc, bonus, r, r["greedy_math500"]))
rows.append("\\midrule\n\\multicolumn{9}{l}{\\textit{Qwen2.5-1.5B-Instruct (second base model)}}\\\\")
for run, lab, sc in (("m2_grpo", "R1", "batch"), ("m2_grpo_gstd", "R1, group std", "group")):
    rows.append(mech_row(lab, sc, "DA", t19d.loc[run], m2_math[run]))
write("tab_collapse", r"""\begin{table*}[!t]
\centering
\caption{Length collapse under group-std advantages. Mean completion length of the training rollouts over steps
61--100 (mid) and 161--200 (late), their ratio, the pre-registered collapse verdict (late $<$ 0.6 $\times$ mid), rollout
accuracy (\%) over steps 161--200, the share (\%) of all-correct groups over steps 61--100 (logged only for the later
runs), and greedy MATH-500 accuracy (\%). DA: difficulty-aware mode reward; length: standard length bonus.}
\label{tab:collapse}
\footnotesize
\begin{tabular}{@{}lllccccccc@{}}
\toprule
Run & Adv.\ scaling & Bonus & Mid & Late & Late/mid & Collapsed & Rollout acc. & All-correct & MATH-500 \\
\midrule
""" + "\n".join(rows).replace("\\multicolumn{9}", "\\multicolumn{10}") + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: LLM judge
t3b = T("table3b_judge_controls")
t3c = T("table3c_judge_onpolicy")
rows = ["\\multicolumn{7}{l}{\\textit{(a) Offline: 1{,}500 base-model answers (750 correct, 750 incorrect)}}\\\\"]
for judge, lab in (("R1-Distill-1.5B · 512 tok (v1 judge)", "R1-Distill-1.5B, 512 tokens"),
                   ("R1-Distill-1.5B · 2048 tok (no cut-off)", "R1-Distill-1.5B, 2{,}048 tokens"),
                   ("Qwen2.5-7B-Instruct · 512 tok (larger)", "Qwen2.5-7B-Instruct, 512 tokens")):
    for parser, plab in (("v1 parser", "lenient parser"), ("corrected parser", "corrected parser")):
        r = t3b[(t3b.judge == judge) & (t3b.parser == parser) & (t3b.subset == "all")].iloc[0]
        rows.append(" & ".join([f"{lab}, {plab}", f"{int(r.n):,}".replace(",", "{,}"), f"{r.precision:.2f}",
                                f"{r.recall:.2f}", f"{r.FPR:.2f} {{\\scriptsize[{r.FPR_ci_low:.2f}, {r.FPR_ci_high:.2f}]}}",
                                f"{r.kappa:.2f}", f"{r.no_valid_score:.2f}"]) + "\\\\")
rows.append("\\midrule\n\\multicolumn{7}{l}{\\textit{(b) On the policies' own greedy answers, R2's judge configuration}}\\\\")
for pol in ("Base (Qwen2.5-Math-1.5B-Inst)", "GRPO ours (R1)", "GRPO judge reward (R2)"):
    r = t3c[(t3c.policy == pol) & (t3c.subset == "all")].iloc[0]
    lab = {"Base (Qwen2.5-Math-1.5B-Inst)": "Base model", "GRPO ours (R1)": "R1", "GRPO judge reward (R2)": "R2"}[pol]
    rows.append(" & ".join([lab, f"{int(r.n):,}".replace(",", "{,}"), f"{r.precision:.2f}", f"{r.recall:.2f}",
                            f"{r.FPR:.2f} {{\\scriptsize[{r.FPR_ci_low:.2f}, {r.FPR_ci_high:.2f}]}}", f"{r.kappa:.2f}",
                            f"{r.no_valid_score:.2f}"]) + "\\\\")
write("tab_judge", r"""\begin{table}[!t]
\centering
\caption{Reliability of small LLM judges against the strict scorer. A score $\ge$ 0.7 counts as acceptance. FPR: share
of incorrect answers accepted (Wilson 95\% interval); No score: share of outputs without a valid score in $[0, 1]$.}
\label{tab:judge}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Judge / answers & $n$ & Prec. & Recall & FPR & $\kappa$ & No score \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: multiple attempts (full benchmarks)
t18a = T("table18a_multiple_attempts")
full = t18a[t18a.vote_source.str.startswith("full") & t18a.bench.isin(["gsm8k", "math500"])]
rows = []
for r in full.itertuples():
    lab = "Base" if r.model.startswith("Base") else "R1"
    rows.append(" & ".join([lab, {"gsm8k": "GSM8K", "math500": "MATH-500"}[r.bench], pct(r.single_greedy),
                            pct(r.any_correct_at_4), pct(r.any_correct_at_K), pct(r.majority_at_4), pct(r.majority_at_K),
                            f"{pct(r.early_stop_vote)} ({100 * r.early_stop_token_saving_vs_maj4:.0f}\\%)",
                            f"{delta(r.maj8_minus_single)} {{\\scriptsize[{delta(r.maj8_minus_single_ci_low)}, {delta(r.maj8_minus_single_ci_high)}]}}",
                            pv(r.maj8_minus_single_p_holm)]) + "\\\\")
write("tab_attempts", r"""\begin{table*}[!t]
\centering
\caption{One answer versus several independent answers (eight samples per problem at temperature 0.7, full
benchmarks). Any@$k$: a correct answer is among the first $k$ samples. Maj@$k$: majority vote with fractional credit on
ties. Early stop: stop after two samples if they agree, else majority of four; in parentheses, tokens saved relative to
maj@4. The last two columns test maj@8 against the greedy answer (paired bootstrap 95\% interval, sign-flip permutation
test, Holm). With 4{,}000 permutation draws, Holm $p$ cannot fall below about 0.006 here.}
\label{tab:attempts}
\footnotesize
\begin{tabular}{@{}llccccccrc@{}}
\toprule
Model & Benchmark & Greedy & Any@4 & Any@8 & Maj@4 & Maj@8 & Early stop & Maj@8 $-$ greedy & Holm $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: breadth vs depth at matched tokens
t18b = T("table18b_breadth_vs_depth")
b = t18b[t18b.vote_source.str.startswith("full") & t18b.bench.isin(["gsm8k", "math500"])]
DEPTH = {"grpo_ours": "R1, forced think", "adaptthink": "AdaptThink-1.5B", "thinkless": "Thinkless-1.5B",
         "r1distill": "R1-Distill-1.5B"}
rows = []
for r in b.itertuples():
    lab = "Base" if r.vote_key == "base" else "R1"
    rows.append(" & ".join([lab, {"gsm8k": "GSM8K", "math500": "MATH-500"}[r.bench], DEPTH[r.depth_key],
                            f"{pct(r.depth_acc)} ({r.depth_tokens:,.0f})".replace(",", "{,}"), str(int(r.k_matched)),
                            f"{pct(r.vote_acc)} ({r.vote_tokens:,.0f})".replace(",", "{,}"),
                            bold_if(f"{delta(r.vote_minus_depth)} {{\\scriptsize[{delta(r.ci_low)}, {delta(r.ci_high)}]}}",
                                    r.p_holm < 0.05), pv(r.p_holm)]) + "\\\\")
write("tab_breadth_depth", r"""\begin{table*}[!t]
\centering
\caption{Second attempts or second thoughts? A majority vote over $k$ independent short answers (breadth) against one
long thinking answer (depth) on the same problems, with $k$ the largest number of votes whose mean token count does
not exceed the thinking answer's. Accuracy (\%) with mean tokens per problem in parentheses. Bold: significant after
Holm correction over the full table (sign-flip permutation test, 4{,}000 draws; Holm $p$ cannot fall below about 0.01
here).}
\label{tab:breadthdepth}
\footnotesize
\begin{tabular}{@{}lllcccrc@{}}
\toprule
Vote of & Benchmark & Thinking answer & Depth acc.\ (tokens) & $k$ & Vote acc.\ (tokens) & Vote $-$ depth & Holm $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: second base model
t19b = T("table19b_second_model_paired")
rows = []
for stage in ("base", "SFT", "GRPO R1 (batch std)", "GRPO R1 (group std)"):
    x = t19a[(t19a.stage == stage) & (t19a.condition == "single")]
    cells = [{"base": "Base", "SFT": "SFT", "GRPO R1 (batch std)": "R1 (batch std)",
              "GRPO R1 (group std)": "R1 (group std)"}[stage]]
    for fam in ("Qwen2.5-Math-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"):
        for bench in ("gsm8k", "math500"):
            r = x[(x.base_model == fam) & (x.bench == bench)].iloc[0]
            cells.append(cell(r.acc, r.ci_low, r.ci_high))
    rows.append(" & ".join(cells) + "\\\\")
verdicts = json.loads((SRC / "table19d_preregistered_verdicts.json").read_text())
note = ("Pre-registered predictions for the second model: (a) the more accurate mode wins "
        f"({'held' if verdicts['a_more_accurate_mode_wins']['held'] else 'failed'}); (b) group std collapses the "
        f"reasoning and batch std does not ({'held' if verdicts['b_group_std_collapses_batch_std_does_not']['held'] else 'failed'}).")
write("tab_second_model", r"""\begin{table*}[!t]
\centering
\caption{The same recipe on two base models: strict accuracy (\%) with Wilson 95\% intervals after each stage
(greedy, own mode choice). """ + note + r"""}
\label{tab:secondmodel}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{Qwen2.5-Math-1.5B-Instruct} & \multicolumn{2}{c}{Qwen2.5-1.5B-Instruct} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Stage & GSM8K & MATH-500 & GSM8K & MATH-500 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- Table: paired comparisons of R1 (notebook Table 2)
t2 = T("table2_paired_comparisons").set_index(["comparison", "bench"])
spec = [("ours − Base (Qwen2.5-Math-1.5B-Inst) [single]", "Base model"),
        ("ours − SFT (v2) [single]", "SFT"), ("ours − DPO (v2) [single]", "DPO"),
        ("ours − GRPO ours (R1) at step 60 [single]", "R1, group std, step 60$^{a}$"),   # adapter of the group-std run
        ("ours − GRPO judge reward (R2) [single]", "R2 (judge reward)"),
        ("ours − GRPO std. length (R3) [single]", "R3 (standard length bonus)"),
        ("ours − GRPO ours, balanced exploration (R1b) [single]", "R1b (balanced exploration)"),
        ("ours − GRPO ours (R1), group-std advantages [single]", "R1 with group-std advantages"),
        ("ours − GRPO ours (R1) [forced_think]", "R1, forced think"),
        ("ours − GRPO ours (R1) [forced_short]", "R1, forced short"),
        ("ours − Thinkless-1.5B (Fang et al.) [single]", "Thinkless-1.5B"),
        ("ours − AdaptThink-1.5B (Zhang et al.) [single]", "AdaptThink-1.5B"),
        ("ours − R1-Distill-1.5B (reference) [single]", "R1-Distill-1.5B")]
n_family = len(t2)
assert n_family == 32, n_family
rows = []
for comp, lab in spec:
    cells = [lab]
    for bench in ("gsm8k", "math500"):
        r = t2.loc[(comp, bench)]
        cells += [bold_if(f"{delta(r.delta)} {{\\scriptsize[{delta(r.ci_low)}, {delta(r.ci_high)}]}}", r.p_holm < 0.05),
                  pv(r.p_mcnemar), pv(r.p_holm)]
    rows.append(" & ".join(cells) + "\\\\")
write("tab_paired", r"""\begin{table*}[!t]
\centering
\caption{R1 (greedy, own mode choice) minus each comparison model on the same problems, in percentage points, with
paired bootstrap 95\% intervals, exact McNemar $p$ and Holm-corrected $p$. The Holm family has """ + str(n_family) +
r""" tests (16 comparisons $\times$ 2 benchmarks); three comparisons are not shown (R1's second seed, R3 with group-std
advantages, and a GRPO model from a preliminary pipeline). Bold: significant after Holm correction.}
\label{tab:paired}
\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{GSM8K ($n=1{,}319$)} & \multicolumn{3}{c}{MATH-500 ($n=500$)} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
R1 minus & $\Delta$ [95\% CI] & $p$ & Holm $p$ & $\Delta$ [95\% CI] & $p$ & Holm $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}\raggedright\footnotesize $^{a}$The step-60 adapter of the run with group-std advantages, saved before
that run's length collapse (Section~\ref{sec:res-collapse}).
\end{table*}
""")

# ---------------------------------------------------------------- Table: MPV against voting (notebook Table 10)
t10 = T("table10_mpv")
rows = []
for model, lab in (("Base (Qwen2.5-Math-1.5B-Inst)", "Base"), ("SFT (v2)", "SFT"), ("DPO (v2)", "DPO"),
                   ("GRPO ours (R1)", "R1")):
    for bench in ("gsm8k", "math500"):
        r = t10[(t10.model == model) & (t10.bench == bench)].iloc[0]
        rows.append(" & ".join([lab, {"gsm8k": "GSM8K", "math500": "MATH-500"}[bench], pct(r.single), pct(r.majority_at_4),
                                pct(r.majority_at_8), pct(r.selection), pct(r.MPV), pct(r.v1_sequential), pct(r.oracle_at_4),
                                f"{r.validator_accuracy:.2f}", f"{r.tokens_MPV:,.0f}".replace(",", "{,}"),
                                f"{r.tokens_majority8:,.0f}".replace(",", "{,}"), pv(r.p_vs_majority8)]) + "\\\\")
write("tab_mpv", r"""\begin{table*}[!t]
\centering
\caption{Multi-perspective reasoning with validation (MPV) against plain voting on the same fixed subsets (100 GSM8K and
100 MATH-500 problems), accuracy (\%). Single: greedy answer. Select: MPV's selected answer before synthesis. Seq.:
sequential revision baseline (run for the base model and R1). Oracle@4: any of MPV's four attempts is correct (an
upper bound). Val.: accuracy of the validation verdicts. Tokens: mean generated tokens per problem. $p$: exact McNemar,
MPV versus maj@8 (uncorrected).}
\label{tab:mpv}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llccccccccccc@{}}
\toprule
Model & Benchmark & Single & Maj@4 & Maj@8 & Select & MPV & Seq. & Oracle@4 & Val. & Tokens MPV & Tokens maj@8 & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
