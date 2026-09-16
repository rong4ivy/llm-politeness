#!/usr/bin/env python3
"""
Full-Dataset Neutral Compression Analysis (Dataset 2)
======================================================
Computes NCI_diff, NCI_ratio, and ISR_diff for all model x prompt cells
on the full Dataset 2 outputs (not only the expert-audit subset).

Confirms that neutral compression is a distributional property of model
predictions across the entire evaluated set (Section 5.2).

Input:  data/dataset2_categorical/llm_predictions.csv
Output: Table printed to stdout
"""

import pandas as pd
import numpy as np

DATA = "data/dataset2_categorical/llm_predictions.csv"
MODELS = ["claude", "deepseek", "gemini", "gpt", "grok"]
PROMPTS = ["p2", "p3", "p4", "p6"]


def col(model, prompt):
    return f"{model}_{prompt}"


def main():
    df = pd.read_csv(DATA).dropna(subset=["human_label"])
    n = len(df)
    print(f"Dataset 2: {n} instances\n")

    pr_r_n = (df["human_label"] == "Neutral").sum() / n
    pr_r_i = (df["human_label"] == "Impolite").sum() / n

    print("=" * 78)
    print("NCI AND ISR ACROSS ALL MODEL x PROMPT CELLS")
    print("=" * 78)
    print(f"Human Neutral proportion: {pr_r_n*100:.1f}%")
    print(f"Human Impolite proportion: {pr_r_i*100:.1f}%")
    print()
    print(f"{'Model':<10} {'Prompt':<8} {'Pr(N)':>8} {'NCI_diff':>10}"
          f" {'NCI_ratio':>10} {'ISR_diff':>10}")
    print("-" * 58)

    nci_diffs, isr_diffs = [], []

    for m in MODELS:
        for p in PROMPTS:
            c = col(m, p)
            if c not in df.columns:
                continue
            valid = df[c].notna()
            sub = df.loc[valid]
            nv = valid.sum()
            if nv < 10:
                continue

            pr_m_n = (sub[c] == "Neutral").sum() / nv
            pr_m_i = (sub[c] == "Impolite").sum() / nv

            nci_d = (pr_m_n - pr_r_n) * 100
            nci_r = pr_m_n / pr_r_n if pr_r_n > 0 else float("inf")
            isr_d = (pr_r_i - pr_m_i) * 100

            nci_diffs.append(nci_d)
            isr_diffs.append(isr_d)

            print(f"{m:<10} {p.upper():<8} {pr_m_n*100:>7.1f}%"
                  f" {nci_d:>+10.1f} {nci_r:>10.2f} {isr_d:>+10.1f}")

    print("-" * 58)
    print(f"{'SUMMARY':<10}")
    print(f"  NCI_diff:  range [{min(nci_diffs):+.1f}, {max(nci_diffs):+.1f}]"
          f"  mean {np.mean(nci_diffs):+.1f} pp")
    print(f"  ISR_diff:  range [{min(isr_diffs):+.1f}, {max(isr_diffs):+.1f}]"
          f"  mean {np.mean(isr_diffs):+.1f} pp")
    print(f"  All NCI_diff > 0: {all(d > 0 for d in nci_diffs)}")
    print(f"  All ISR_diff > 0: {all(d > 0 for d in isr_diffs)}")


if __name__ == "__main__":
    main()
