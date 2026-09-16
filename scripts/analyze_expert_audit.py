#!/usr/bin/env python3
"""
Expert Audit Analysis
======================
Reproduces the expert-audit results (Section 5.3, Table 1, Appendix F):
  - Expert inter-annotator agreement (Fleiss' kappa, Krippendorff's alpha)
  - Mean model-level LLM vs. expert consensus
  - Mean model-level LLM vs. crowd labels
  - Crowd vs. expert consensus
  - Neutral compression diagnostics on the audit subset
  - Model-level breakdown (Appendix F, Panel A)

Input:  data/expert_audit/expert_llm_crowd_annotations.csv
Output: Tables printed to stdout

Note: The prompt-consensus label for each model is derived by majority
vote across four prompt conditions.  Reported LLM--expert and LLM--crowd
statistics are unweighted means of the five model-level results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from itertools import combinations
from collections import Counter

DATA = "data/expert_audit/expert_llm_crowd_annotations.csv"
MODELS = ["claude", "deepseek", "gemini", "gpt", "grok"]
EXPERTS = ["Zhang", "Luo", "Chen", "Jiang", "Li"]
PROMPTS = ["p2", "p3", "p4", "p6"]
LABELS = ["Impolite", "Neutral", "Polite"]


def fleiss_kappa(rating_matrix, n_raters):
    n_samples, n_cats = rating_matrix.shape
    p_j = np.sum(rating_matrix, axis=0) / (n_samples * n_raters)
    P_i = (np.sum(rating_matrix ** 2, axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0


def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} expert-audited instances\n")

    # ==================================================================
    # 1. Expert inter-annotator agreement
    # ==================================================================
    print("=" * 78)
    print("1. EXPERT INTER-ANNOTATOR AGREEMENT")
    print("=" * 78)

    # Pairwise Cohen's kappa
    kappas = []
    for e1, e2 in combinations(EXPERTS, 2):
        c1, c2 = f"{e1}_label", f"{e2}_label"
        valid = df[c1].notna() & df[c2].notna()
        if valid.sum() > 0:
            k = cohen_kappa_score(df.loc[valid, c1], df.loc[valid, c2])
            kappas.append(k)
    print(f"Mean pairwise Cohen's kappa: {np.mean(kappas):.3f}"
          f"  (SD: {np.std(kappas):.3f})")

    # Fleiss' kappa
    rating_mat = np.zeros((len(df), len(LABELS)))
    for _, row in df.iterrows():
        for e in EXPERTS:
            lbl = row[f"{e}_label"]
            if pd.notna(lbl) and lbl in LABELS:
                rating_mat[_, LABELS.index(lbl)] += 1
    # rebuild properly with index
    rating_mat = np.zeros((len(df), len(LABELS)))
    for i, (_, row) in enumerate(df.iterrows()):
        for e in EXPERTS:
            lbl = row[f"{e}_label"]
            if pd.notna(lbl) and lbl in LABELS:
                rating_mat[i, LABELS.index(lbl)] += 1
    fk = fleiss_kappa(rating_mat, len(EXPERTS))
    print(f"Fleiss' kappa: {fk:.3f}")

    consensus_dist = df["agreement_level"].value_counts().sort_index(
        ascending=False
    )
    high_consensus = (df["agreement_level"] >= 4).sum()
    print(f"Strong/full consensus (>=4/5): {high_consensus}/{len(df)}"
          f" ({high_consensus/len(df)*100:.1f}%)")

    # ==================================================================
    # 2. Model-level agreement with expert consensus (Appendix F Panel A)
    # ==================================================================
    print()
    print("=" * 78)
    print("2. MODEL-LEVEL AGREEMENT WITH EXPERT CONSENSUS  (Panel A)")
    print("=" * 78)
    print(f"{'Model':<10} {'Acc':>8} {'Kappa':>8} {'Macro-F1':>10}"
          f" {'Prompt kappa':>13}")
    print("-" * 51)

    model_accs, model_kappas, model_f1s = [], [], []

    for m in MODELS:
        cons_col = f"{m}_consensus"
        valid = df[cons_col].notna()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, "majority_label"]
        y_pred = df.loc[valid, cons_col]

        acc = accuracy_score(y_true, y_pred)
        kap = cohen_kappa_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS)
        model_accs.append(acc)
        model_kappas.append(kap)
        model_f1s.append(f1)

        # Prompt consistency: pairwise kappa across 4 prompts
        prompt_kappas = []
        for p1, p2 in combinations(PROMPTS, 2):
            c1, c2 = f"{m}_{p1}", f"{m}_{p2}"
            if c1 in df.columns and c2 in df.columns:
                v = df[c1].notna() & df[c2].notna()
                if v.sum() > 0:
                    pk = cohen_kappa_score(df.loc[v, c1], df.loc[v, c2])
                    prompt_kappas.append(pk)
        pk_mean = np.mean(prompt_kappas) if prompt_kappas else float("nan")

        print(f"{m:<10} {acc:>8.3f} {kap:>8.3f} {f1:>10.3f}"
              f" {pk_mean:>13.3f}")

    print("-" * 51)
    print(f"{'Mean':<10} {np.mean(model_accs):>8.3f}"
          f" {np.mean(model_kappas):>8.3f}"
          f" {np.mean(model_f1s):>10.3f}")

    # ==================================================================
    # 3. Mean model-level LLM vs. crowd & crowd vs. expert
    # ==================================================================
    print()
    print("=" * 78)
    print("3. AGREEMENT COMPARISONS  (Table 1, Panel A)")
    print("=" * 78)

    # LLM vs crowd (per model, then average)
    lc_accs, lc_kappas = [], []
    for m in MODELS:
        cons_col = f"{m}_consensus"
        valid = df[cons_col].notna() & df["human_label"].notna()
        if valid.sum() == 0:
            continue
        lc_accs.append(accuracy_score(
            df.loc[valid, "human_label"], df.loc[valid, cons_col]))
        lc_kappas.append(cohen_kappa_score(
            df.loc[valid, "human_label"], df.loc[valid, cons_col]))

    print(f"Mean LLM vs. Expert:  Acc={np.mean(model_accs):.3f},"
          f"  kappa={np.mean(model_kappas):.3f},"
          f"  F1={np.mean(model_f1s):.3f}")
    print(f"Mean LLM vs. Crowd:   Acc={np.mean(lc_accs):.3f},"
          f"  kappa={np.mean(lc_kappas):.3f}")

    # Crowd vs Expert
    valid = df["human_label"].notna() & df["majority_label"].notna()
    ce_acc = accuracy_score(
        df.loc[valid, "human_label"], df.loc[valid, "majority_label"])
    ce_kap = cohen_kappa_score(
        df.loc[valid, "human_label"], df.loc[valid, "majority_label"])
    ce_f1 = f1_score(
        df.loc[valid, "human_label"], df.loc[valid, "majority_label"],
        average="macro", labels=LABELS)
    print(f"Crowd vs. Expert:     Acc={ce_acc:.3f},"
          f"  kappa={ce_kap:.3f},"
          f"  F1={ce_f1:.3f}")

    # ==================================================================
    # 4. Label distributions (Table 1, Panel B)
    # ==================================================================
    print()
    print("=" * 78)
    print("4. LABEL DISTRIBUTIONS  (Table 1, Panel B)")
    print("=" * 78)

    n = len(df)
    print(f"{'Source':<18} {'Impolite':>10} {'Neutral':>10} {'Polite':>10}")
    print("-" * 50)

    for name, series in [("Crowd", df["human_label"]),
                         ("Expert consensus", df["majority_label"])]:
        dist = series.value_counts()
        print(f"{name:<18}", end="")
        for lbl in LABELS:
            pct = dist.get(lbl, 0) / n * 100
            print(f" {pct:>9.1f}%", end="")
        print()

    # Mean LLM distribution
    llm_dist = {lbl: 0 for lbl in LABELS}
    for m in MODELS:
        cons = df[f"{m}_consensus"].dropna()
        dist = cons.value_counts()
        for lbl in LABELS:
            llm_dist[lbl] += dist.get(lbl, 0)
    total = sum(llm_dist.values())
    print(f"{'Mean LLM':<18}", end="")
    for lbl in LABELS:
        print(f" {llm_dist[lbl]/total*100:>9.1f}%", end="")
    print()

    # ==================================================================
    # 5. Neutral compression diagnostics (Table 1, Panel C)
    # ==================================================================
    print()
    print("=" * 78)
    print("5. NEUTRAL COMPRESSION DIAGNOSTICS  (Table 1, Panel C)")
    print("=" * 78)

    # Compute against expert consensus
    pr_m_n = llm_dist["Neutral"] / total
    pr_e_n = (df["majority_label"] == "Neutral").sum() / n
    pr_c_n = (df["human_label"] == "Neutral").sum() / n
    pr_m_i = llm_dist["Impolite"] / total
    pr_e_i = (df["majority_label"] == "Impolite").sum() / n
    pr_c_i = (df["human_label"] == "Impolite").sum() / n

    print(f"{'Metric':<16} {'vs Expert':>12} {'vs Crowd':>12}")
    print("-" * 42)
    print(f"{'NCI_diff (pp)':<16} {(pr_m_n - pr_e_n)*100:>+12.1f}"
          f" {(pr_m_n - pr_c_n)*100:>+12.1f}")
    print(f"{'NCI_ratio':<16} {pr_m_n/pr_e_n if pr_e_n else 0:>12.2f}"
          f" {pr_m_n/pr_c_n if pr_c_n else 0:>12.2f}")
    print(f"{'ISR_diff (pp)':<16} {(pr_e_i - pr_m_i)*100:>+12.1f}"
          f" {(pr_c_i - pr_m_i)*100:>+12.1f}")


if __name__ == "__main__":
    main()
