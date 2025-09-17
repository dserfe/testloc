import argparse, json, sys, os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_occurrence_from_patch_jsonl(patch_path: str) -> int:
    """
    Reads the 'occurrence' (majority vote) signal from a patch .jsonl file.
    - Supports keys: 'occurrence', 'occurrences', 'votes'
    - If multiple lines exist, returns the maximum value found.
    - If file missing or malformed, returns 0.
    """
    occ = 0
    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                for k in ("occurrence", "occurrences", "votes"):
                    if k in obj:
                        try:
                            occ = max(occ, int(obj[k]))
                        except Exception:
                            pass
    except FileNotFoundError:
        # Keep 0; caller may log if desired
        pass
    except Exception:
        # Be robust: treat as 0 if unreadable
        pass
    return occ

def aggregate_repro(summary: dict) -> Tuple[Dict[str,int], int]:
    """
    Returns:
      repro_flips_per_patch: {patch_file: num_repro_tests_fixed}
      total_repro_considered: number of reproduction tests with repro_fails_on_clean==True
    """
    repro = summary.get("reproduction", {}) or {}
    flips: Dict[str, int] = defaultdict(int)
    total_considered = 0

    # Initialize with regression patch files so that 0-flip patches still appear
    reg = summary.get("regression", {}) or {}
    for r in reg.get("results", []) or []:
        pf = r.get("patch_file")
        if pf:
            flips.setdefault(pf, 0)

    for test_block in repro.values():
        if not isinstance(test_block, dict):
            continue
        if not test_block.get("repro_fails_on_clean", False):
            continue
        total_considered += 1
        for oc in (test_block.get("outcomes", []) or []):
            pf = oc.get("patch_file")
            if pf and bool(oc.get("passed")):
                flips[pf] = flips.get(pf, 0) + 1

    return dict(flips), total_considered

def aggregate_regression(summary: dict) -> Dict[str, int]:
    """#passed regression tests per patch_file."""
    reg = summary.get("regression", {}) or {}
    results = reg.get("results", []) or []
    return {
        r.get("patch_file"): int(r.get("passed", 0) or 0)
        for r in results
        if r.get("patch_file")
    }

def collect_patch_files(summary: dict) -> List[str]:
    """Union of patch files in first-seen order (no filename tiebreak)."""
    seen = set()
    ordered: List[str] = []

    reg = summary.get("regression", {}) or {}
    for r in reg.get("results", []) or []:
        pf = r.get("patch_file")
        if pf and pf not in seen:
            seen.add(pf)
            ordered.append(pf)

    repro = summary.get("reproduction", {}) or {}
    for tb in repro.values():
        for oc in (tb.get("outcomes", []) or []):
            pf = oc.get("patch_file")
            if pf and pf not in seen:
                seen.add(pf)
                ordered.append(pf)

    return ordered


# def collect_patch_files(summary: dict) -> List[str]:
#     """Union of patch files in first-seen order (no filename tiebreak)."""
#     seen = set()
#     ordered: List[str] = []

#     reg = summary.get("regression", {}) or {}
#     for r in reg.get("results", []) or []:
#         pf = r.get("patch_file")
#         if pf and pf not in seen:
#             seen.add(pf)
#             ordered.append(pf)

#     repro = summary.get("reproduction", {}) or {}
#     for tb in repro.values():
#         for oc in (tb.get("outcomes", []) or []):
#             pf = oc.get("patch_file")
#             if pf and pf not in seen:
#                 seen.add(pf)
#                 ordered.append(pf)

#     return ordered 
# def rank_patches(summary: dict) -> List[dict]:
#     repro_flips, total_repro = aggregate_repro(summary)
#     reg_pass = aggregate_regression(summary)
#     patch_files = collect_patch_files(summary)

#     ranked = []
#     for pf in patch_files:
#         ranked.append({
#             "patch_file": pf,
#             "repro_fixes": repro_flips.get(pf, 0),
#             "repro_total_considered": total_repro,
#             "regression_passed": reg_pass.get(pf, 0),
#             "occurrence": read_occurrence_from_patch_jsonl(pf),
#         })

#     # Fallback ranking:
#     # 1) If ANY repro_fixes > 0, rank by repro_fixes (desc),
#     #    then use regression_passed and occurrence only as tie-breakers.
#     # 2) Else, if ANY regression_passed > 0, rank by regression_passed (desc),
#     #    then break ties with occurrence.
#     # 3) Else, rank by occurrence only.
#     max_repro = max((d["repro_fixes"] for d in ranked), default=0)
#     if max_repro > 0:
#         ranked.sort(key=lambda d: (-d["repro_fixes"],
#                                    -d["regression_passed"],
#                                    -d["occurrence"]))
#         return ranked

#     max_reg = max((d["regression_passed"] for d in ranked), default=0)
#     if max_reg > 0:
#         ranked.sort(key=lambda d: (-d["regression_passed"],
#                                    -d["occurrence"]))
#         return ranked

#     ranked.sort(key=lambda d: (-d["occurrence"],))
#     return ranked

import random
from typing import List, Dict

# def rank_patches(summary: dict, seed: int | None = None) -> List[Dict]:
#     # Optional reproducibility for tie-breaking randomness
#     if seed is not None:
#         random.seed(seed)

#     repro_flips, total_repro = aggregate_repro(summary)
#     reg_pass = aggregate_regression(summary)
#     patch_files = collect_patch_files(summary)

#     ranked = []
#     for pf in patch_files:
#         ranked.append({
#             "patch_file": pf,
#             "repro_fixes": repro_flips.get(pf, 0),
#             "repro_total_considered": total_repro,
#             "regression_passed": reg_pass.get(pf, 0),
#             # keep occurrence in the record if you still want to inspect it,
#             # but it is NOT used for ranking anymore:
#             "occurrence": read_occurrence_from_patch_jsonl(pf),
#         })

#     max_repro = max((d["repro_fixes"] for d in ranked), default=0)
#     if max_repro > 0:
#         # Sort by reproduction fixes desc, then regression passes desc,
#         # then randomize ties.
#         # We inject a random number as the last key so equal (repro, reg) get shuffled.
#         ranked.sort(key=lambda d: (-d["repro_fixes"],
#                                    -d["regression_passed"],
#                                    random.random()))
#         return ranked

#     max_reg = max((d["regression_passed"] for d in ranked), default=0)
#     if max_reg > 0:
#         # No reproduction fixes > 0 anywhere; sort by regression passes,
#         # randomize ties.
#         ranked.sort(key=lambda d: (-d["regression_passed"], random.random()))
#         return ranked

#     # No signal from reproduction or regression; return in a random order.
#     random.shuffle(ranked)
#     return ranked


import random
from typing import List, Dict

def rank_patches(summary: dict, seed: int | None = None) -> List[Dict]:
    """
    Primary: reproduction fixes
    Secondary: regression passes
    Tertiary: occurrence
    Final tie-breaker: random (seedable)
    """
    if seed is not None:
        random.seed(seed)

    repro_flips, total_repro = aggregate_repro(summary)
    reg_pass = aggregate_regression(summary)
    patch_files = collect_patch_files(summary)

    ranked: List[Dict] = []
    for pf in patch_files:
        item = {
            "patch_file": pf,
            "repro_fixes": repro_flips.get(pf, 0),
            "repro_total_considered": total_repro,
            "regression_passed": reg_pass.get(pf, 0),
            "occurrence": read_occurrence_from_patch_jsonl(pf),
            # precompute a random key so sort is stable and reproducible with seed
            "_rand": random.random(),
        }
        ranked.append(item)

    # If there exists any nonzero repro_fixes, use the full cascade.
    max_repro = max((d["repro_fixes"] for d in ranked), default=0)
    if max_repro > 0:
        ranked.sort(key=lambda d: (-d["repro_fixes"],
                                   -d["regression_passed"],
                                   -d["occurrence"],
                                   d["_rand"]))
        for d in ranked:
            d.pop("_rand", None)
        return ranked

    # Otherwise, fall back to regression -> occurrence -> random.
    max_reg = max((d["regression_passed"] for d in ranked), default=0)
    if max_reg > 0:
        ranked.sort(key=lambda d: (-d["regression_passed"],
                                #    -d["occurrence"],
                                #    d["_rand"]
                                   ))
        for d in ranked:
            d.pop("_rand", None)
        return ranked

    # No signal from reproduction or regression; use occurrence -> random.
    ranked.sort(key=lambda d: (-d["occurrence"], d["_rand"]))
    for d in ranked:
        d.pop("_rand", None)
    return ranked


def pick_top(ranked: List[dict]) -> dict:
    return ranked[0] if ranked else {}

def main():
    ap = argparse.ArgumentParser(
        description="Rank patches by (1) repro fail→pass flips, (2) regression passed, (3) majority vote from each patch's .jsonl 'occurrence'."
    )
    ap.add_argument("--summary_json", help="Path to the per-instance summary JSON.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, also output top-k list alongside the winner.")
    ap.add_argument("--out", help="Write result JSON to a file instead of stdout.")
    args = ap.parse_args()

    summary = load_json(args.summary_json)
    ranked = rank_patches(summary)
    winner = pick_top(ranked)

    output = {
        "instance_id": summary.get("instance_id"),
        "selected_patch": winner,
        "ranked_patches": ranked[:args.topk] if args.topk and args.topk > 0 else ranked
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)

if __name__ == "__main__":
    main()
