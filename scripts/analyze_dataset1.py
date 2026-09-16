#!/usr/bin/env python3
"""
Dataset 1 Analysis: Continuous Politeness Scoring
==================================================
Reproduces the main results for Dataset 1 (Section 5.1):
  - Model-human alignment (Pearson r, MAE, Close Rate)
  - Inter-model vs. model-human correlations
  - Prompt-effect ANOVA

Input:  data/dataset1_continuous/llm_scores.csv
Output: Tables printed to stdout
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations

DATA = "data/dataset1_continuous/llm_scores.csv"
MODELS = ["claude", "deepseek", "gemini", "gpt", "grok"]
PROMPTS = ["p1", "p2", "p3", "p4", "p5", "p6"]


def col(model, prompt):
    return f"{model}_{prompt}"


def main():
    df = pd.read_csv(DATA)
    human = df["human_score"].values
    print(f"Loaded {len(df)} instances, {len(df.columns)} columns\n")

    # ==================================================================
    # 1. Model-human alignment (Table in paper: Close Rate, MAE, r)
    # ==================================================================
    print("=" * 78)
    print("1. MODEL-HUMAN ALIGNMENT  (averaged across prompts)")
    print("=" * 78)
    print(f"{'Model':<10} {'Pearson r':>10} {'MAE':>10} {'Close Rate':>12}")
    print("-" * 44)

    for m in MODELS:
        rs, maes, crs = [], [], []
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            s = df[c].values
            mask = ~(np.isnan(s) | np.isnan(human))
            h, ms_ = human[mask], s[mask]
            r, _ = pearsonr(h, ms_)
            mae = np.mean(np.abs(h - ms_))
            cr = np.mean(np.abs(h - ms_) < 0.5)
            rs.append(r)
            maes.append(mae)
            crs.append(cr)
        print(f"{m:<10} {np.mean(rs):>10.3f} {np.mean(maes):>10.3f}"
              f" {np.mean(crs):>12.3f}")

    # ==================================================================
    # 2. Inter-model vs. model-human correlations
    # ==================================================================
    print()
    print("=" * 78)
    print("2. INTER-MODEL vs. MODEL-HUMAN CORRELATIONS  (per prompt)")
    print("=" * 78)
    print(f"{'Prompt':<8} {'Model-Human':>13} {'Inter-Model':>13} {'Gap':>8}")
    print("-" * 44)

    all_mh, all_im = [], []
    for p in PROMPTS:
        mh = []
        for m in MODELS:
            c = col(m, p)
            if c not in df.columns:
                continue
            s = df[c].values
            mask = ~(np.isnan(s) | np.isnan(human))
            r, _ = pearsonr(human[mask], s[mask])
            mh.append(r)

        im = []
        for m1, m2 in combinations(MODELS, 2):
            c1, c2 = col(m1, p), col(m2, p)
            if c1 not in df.columns or c2 not in df.columns:
                continue
            s1, s2 = df[c1].values, df[c2].values
            mask = ~(np.isnan(s1) | np.isnan(s2))
            r, _ = pearsonr(s1[mask], s2[mask])
            im.append(r)

        all_mh.extend(mh)
        all_im.extend(im)
        print(f"{p.upper():<8} {np.mean(mh):>13.4f} {np.mean(im):>13.4f}"
              f" {np.mean(im) - np.mean(mh):>+8.4f}")

    print("-" * 44)
    print(f"{'Overall':<8} {np.mean(all_mh):>13.4f} {np.mean(all_im):>13.4f}"
          f" {np.mean(all_im) - np.mean(all_mh):>+8.4f}")

    # ==================================================================
    # 3. Pairwise inter-model correlation matrix (avg across prompts)
    # ==================================================================
    print()
    print("=" * 78)
    print("3. PAIRWISE CORRELATION MATRIX  (averaged across 6 prompts)")
    print("=" * 78)

    pair_avg = {}
    for m1, m2 in combinations(MODELS, 2):
        rs = []
        for p in PROMPTS:
            c1, c2 = col(m1, p), col(m2, p)
            if c1 not in df.columns or c2 not in df.columns:
                continue
            s1, s2 = df[c1].values, df[c2].values
            mask = ~(np.isnan(s1) | np.isnan(s2))
            r, _ = pearsonr(s1[mask], s2[mask])
            rs.append(r)
        pair_avg[(m1, m2)] = np.mean(rs)

    mh_avg = {}
    for m in MODELS:
        rs = []
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            s = df[c].values
            mask = ~(np.isnan(s) | np.isnan(human))
            r, _ = pearsonr(human[mask], s[mask])
            rs.append(r)
        mh_avg[m] = np.mean(rs)

    header = f"{'':>10}" + "".join(f"{m:>10}" for m in MODELS) + f"{'human':>10}"
    print(header)
    for m1 in MODELS:
        row = f"{m1:>10}"
        for m2 in MODELS:
            if m1 == m2:
                row += f"{'1.000':>10}"
            else:
                key = (m1, m2) if (m1, m2) in pair_avg else (m2, m1)
                row += f"{pair_avg[key]:>10.3f}"
        row += f"{mh_avg[m1]:>10.3f}"
        print(row)
    print(f"{'human':>10}" + "".join(f"{mh_avg[m]:>10.3f}" for m in MODELS)
          + f"{'1.000':>10}")

    # ==================================================================
    # 4. Prompt-effect ANOVA
    # ==================================================================
    print()
    print("=" * 78)
    print("4. PROMPT-EFFECT ANOVA")
    print("=" * 78)

    from scipy import stats as sp_stats
    cell_r = []
    for p in PROMPTS:
        for m in MODELS:
            c = col(m, p)
            if c not in df.columns:
                continue
            s = df[c].values
            mask = ~(np.isnan(s) | np.isnan(human))
            r, _ = pearsonr(human[mask], s[mask])
            cell_r.append({"prompt": p, "model": m, "r": r})

    df_r = pd.DataFrame(cell_r)
    groups = [df_r[df_r["prompt"] == p]["r"].values for p in PROMPTS]
    F, pval = sp_stats.f_oneway(*groups)
    print(f"One-way ANOVA on r:  F({len(PROMPTS)-1}, "
          f"{len(df_r)-len(PROMPTS)}) = {F:.4f},  p = {pval:.4f}")

    try:
        from statsmodels.formula.api import ols
        import statsmodels.api as sm

        fit = ols("r ~ C(prompt) + C(model)", data=df_r).fit()
        tbl = sm.stats.anova_lm(fit, typ=2)
        ss_p = tbl.loc["C(prompt)", "sum_sq"]
        ss_m = tbl.loc["C(model)", "sum_sq"]
        ss_r = tbl.loc["Residual", "sum_sq"]
        ss_t = ss_p + ss_m + ss_r
        print(f"Two-way ANOVA:  prompt {ss_p/ss_t*100:.1f}%,"
              f"  model {ss_m/ss_t*100:.1f}%,"
              f"  residual {ss_r/ss_t*100:.1f}%")
        print(f"  Prompt F = {tbl.loc['C(prompt)','F']:.4f},"
              f"  p = {tbl.loc['C(prompt)','PR(>F)']:.4f}")
        print(f"  Model  F = {tbl.loc['C(model)','F']:.4f},"
              f"  p = {tbl.loc['C(model)','PR(>F)']:.4f}")
    except ImportError:
        print("(Install statsmodels for two-way ANOVA)")


if __name__ == "__main__":
    main()
