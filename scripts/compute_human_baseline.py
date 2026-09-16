#!/usr/bin/env python3
"""
Human Baseline Estimation (Appendix B)
========================================
Computes inter-annotator agreement and leave-one-annotator-out
human reference for Dataset 1 (Wikipedia subset).

  - Pairwise Pearson r among 5 annotators
  - Krippendorff's alpha (interval)
  - ICC(A,1) and ICC(A,k)
  - Leave-one-out: each annotator vs. 4-person consensus

Input:  data/dataset1_continuous/iaa_results.json (precomputed)
        data/dataset1_continuous/human_ceiling.json (precomputed)
Output: Summary printed to stdout

To recompute from raw annotations, install convokit and run
extract_raw_annotations.py first.
"""

import json
import os

IAA_FILE = "data/dataset1_continuous/iaa_results.json"
CEIL_FILE = "data/dataset1_continuous/human_ceiling.json"


def main():
    print("=" * 78)
    print("HUMAN BASELINE ANALYSIS  (Dataset 1, Wikipedia subset)")
    print("=" * 78)

    if os.path.exists(IAA_FILE):
        with open(IAA_FILE) as f:
            iaa = json.load(f)
        pr = iaa["pairwise_pearson"]
        print(f"\nInter-annotator agreement ({iaa['n_utterances']} utterances,"
              f" {iaa['n_annotators']} annotators):")
        print(f"  Pairwise Pearson r: {pr['mean']:.4f}"
              f"  (SD: {pr['std']:.4f})")
        if iaa.get("krippendorff_alpha") is not None:
            print(f"  Krippendorff's alpha: {iaa['krippendorff_alpha']:.4f}")
        icc = iaa.get("icc", {})
        if icc.get("icc2_1_single_rater") is not None:
            print(f"  ICC(A,1) single rater: {icc['icc2_1_single_rater']:.4f}")
            print(f"  ICC(A,k) average rater:"
                  f" {icc['icc2_k_average_rater']:.4f}")
        vs = iaa.get("variance_statistics", {})
        if vs:
            total = vs.get("total_variance", 1)
            within = vs.get("mean_within_utterance_variance", 0)
            between = total - within
            print(f"  Within-utterance variance: {within/total*100:.1f}%")
            print(f"  Between-utterance variance: {between/total*100:.1f}%")
    else:
        print(f"  [!] {IAA_FILE} not found")

    if os.path.exists(CEIL_FILE):
        with open(CEIL_FILE) as f:
            ceil = json.load(f)
        print(f"\nLeave-one-annotator-out reference:")
        for key in ["pearson_r", "spearman_r", "mae_original", "rmse_original",
                     "accuracy_3class", "f1_macro_3class", "kappa_3class"]:
            if key in ceil:
                print(f"  {key}: {ceil[key]:.4f}")
    else:
        print(f"  [!] {CEIL_FILE} not found")


if __name__ == "__main__":
    main()
