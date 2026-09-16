#!/usr/bin/env python3
"""
Dataset 2 Analysis: Three-way Politeness Classification
========================================================
Reproduces the main results for Dataset 2 (Section 5.2):
  - Model-human agreement (Accuracy, Cohen's kappa, Macro-F1)
  - Inter-model agreement (Fleiss' kappa per prompt)
  - Neutral compression diagnostics (NCI, ISR, ENSR)

Input:  data/dataset2_categorical/llm_predictions.csv
Output: Tables printed to stdout
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from collections import Counter

DATA = "data/dataset2_categorical/llm_predictions.csv"
MODELS = ["claude", "deepseek", "gemini", "gpt", "grok"]
PROMPTS = ["p2", "p3", "p4", "p6"]
LABELS = ["Impolite", "Neutral", "Polite"]


def col(model, prompt):
    return f"{model}_{prompt}"


def fleiss_kappa(rating_matrix, n_raters):
    """Fleiss' kappa for multi-rater categorical agreement."""
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
    df = df.dropna(subset=["human_label"])
    print(f"Loaded {len(df)} instances\n")

    # ==================================================================
    # 1. Model-human agreement per model (averaged across prompts)
    # ==================================================================
    print("=" * 78)
    print("1. MODEL-HUMAN AGREEMENT  (averaged across prompts)")
    print("=" * 78)
    print(f"{'Model':<10} {'Accuracy':>10} {'Kappa':>10} {'Macro-F1':>10}")
    print("-" * 42)

    for m in MODELS:
        accs, kappas, f1s = [], [], []
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            valid = df[c].notna()
            if valid.sum() < 10:
                continue
            y_true = df.loc[valid, "human_label"]
            y_pred = df.loc[valid, c]
            accs.append(accuracy_score(y_true, y_pred))
            kappas.append(cohen_kappa_score(y_true, y_pred))
            f1s.append(f1_score(y_true, y_pred, average="macro",
                                labels=LABELS))
        if accs:
            print(f"{m:<10} {np.mean(accs):>10.3f} {np.mean(kappas):>10.3f}"
                  f" {np.mean(f1s):>10.3f}")

    # ==================================================================
    # 2. Inter-model agreement (Fleiss' kappa per prompt)
    # ==================================================================
    print()
    print("=" * 78)
    print("2. INTER-MODEL AGREEMENT  (Fleiss' kappa per prompt)")
    print("=" * 78)
    print(f"{'Prompt':<8} {'Fleiss kappa':>13} {'N valid':>10}")
    print("-" * 33)

    for p in PROMPTS:
        model_cols = [col(m, p) for m in MODELS if col(m, p) in df.columns]
        sub = df[model_cols].dropna()
        n = len(sub)
        if n == 0:
            continue

        rating_mat = np.zeros((n, len(LABELS)))
        for i, (_, row) in enumerate(sub.iterrows()):
            for mc in model_cols:
                lbl = row[mc]
                if lbl in LABELS:
                    rating_mat[i, LABELS.index(lbl)] += 1

        fk = fleiss_kappa(rating_mat, len(model_cols))
        print(f"{p.upper():<8} {fk:>13.3f} {n:>10}")

    # ==================================================================
    # 3. Neutral compression diagnostics (NCI, ISR, ENSR)
    # ==================================================================
    print()
    print("=" * 78)
    print("3. NEUTRAL COMPRESSION DIAGNOSTICS")
    print("=" * 78)
    print(f"{'Model':<10} {'Prompt':<8} {'NCI_diff':>10} {'NCI_ratio':>10}"
          f" {'ISR_diff':>10}")
    print("-" * 50)

    for m in MODELS:
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            valid = df[c].notna()
            if valid.sum() < 10:
                continue
            y_true = df.loc[valid, "human_label"]
            y_pred = df.loc[valid, c]
            n = valid.sum()

            pr_m_n = (y_pred == "Neutral").sum() / n
            pr_r_n = (y_true == "Neutral").sum() / n
            pr_m_i = (y_pred == "Impolite").sum() / n
            pr_r_i = (y_true == "Impolite").sum() / n

            nci_diff = (pr_m_n - pr_r_n) * 100
            nci_ratio = pr_m_n / pr_r_n if pr_r_n > 0 else float("inf")
            isr_diff = (pr_r_i - pr_m_i) * 100

            print(f"{m:<10} {p.upper():<8} {nci_diff:>+10.1f}"
                  f" {nci_ratio:>10.2f} {isr_diff:>+10.1f}")

    # ==================================================================
    # 4. Label distribution summary
    # ==================================================================
    print()
    print("=" * 78)
    print("4. LABEL DISTRIBUTION SUMMARY  (mean across models & prompts)")
    print("=" * 78)

    human_dist = df["human_label"].value_counts(normalize=True)
    print("Human reference:")
    for lbl in LABELS:
        print(f"  {lbl:<10} {human_dist.get(lbl, 0)*100:>6.1f}%")

    print("\nMean LLM distribution:")
    llm_counts = {lbl: 0 for lbl in LABELS}
    n_cells = 0
    for m in MODELS:
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            valid = df[c].notna()
            dist = df.loc[valid, c].value_counts(normalize=True)
            for lbl in LABELS:
                llm_counts[lbl] += dist.get(lbl, 0)
            n_cells += 1

    for lbl in LABELS:
        print(f"  {lbl:<10} {llm_counts[lbl] / n_cells * 100:>6.1f}%")


if __name__ == "__main__":
    main()
