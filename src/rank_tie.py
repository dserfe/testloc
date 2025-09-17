import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_occurrence_from_patch_jsonl(patch_path: str) -> int:
    """Reads 'occurrence' field from patch JSONL if available."""
    occ = 0
    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    for k in ("occurrence", "occurrences", "votes"):
                        if k in obj:
                            occ = max(occ, int(obj[k]))
                except Exception:
                    continue
    except Exception:
        pass
    return occ


def aggregate_repro(summary: dict) -> Tuple[Dict[str, int], int]:
    repro = summary.get("reproduction", {}) or {}
    flips: Dict[str, int] = defaultdict(int)
    total_considered = 0

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
    reg = summary.get("regression", {}) or {}
    results = reg.get("results", []) or []
    return {
        r.get("patch_file"): int(r.get("passed", 0) or 0)
        for r in results
        if r.get("patch_file")
    }


def collect_patch_files(summary: dict) -> List[str]:
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


def rank_patches(summary: dict, seed: Optional[int] = None) -> List[List[Dict]]:
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
            "_rand": random.random(),
        }
        ranked.append(item)

    max_repro = max((d["repro_fixes"] for d in ranked), default=0)

    # Primary key: reproduction fixes, then regression passed
    if max_repro > 0:
        key_func = lambda d: (-d["repro_fixes"], -d["regression_passed"])
    else:
        key_func = lambda d: (-d["regression_passed"],)

    # Sort: primary key, then occurrence, then random
    ranked.sort(key=lambda d: (*key_func(d), -d["occurrence"], d["_rand"]))

    # Group by primary key only (ignoring occurrence/random for group definition)
    tie_groups: List[List[Dict]] = []
    current_group = []
    current_key = None

    for d in ranked:
        d.pop("_rand", None)
        k = key_func(d)
        if k != current_key:
            if current_group:
                tie_groups.append(current_group)
            current_group = [d]
            current_key = k
        else:
            current_group.append(d)

    if current_group:
        tie_groups.append(current_group)

    return tie_groups


def pick_top(ranked_groups: List[List[dict]], seed: Optional[int] = None, pick_one: bool = False) -> Any:
    if not ranked_groups or not ranked_groups[0]:
        return None if pick_one else []

    top_group = ranked_groups[0]
    if pick_one:
        # Break ties by occurrence
        max_occ = max(d.get("occurrence", 0) for d in top_group)
        best = [d for d in top_group if d.get("occurrence", 0) == max_occ]
        # Break ties randomly
        if seed is not None:
            random.seed(seed)
        return random.choice(best) if len(best) > 1 else best[0]
    else:
        return top_group


def group_with_rank_keys(ranked_groups: List[List[dict]], topk: Optional[int] = None) -> Dict[str, List[dict]]:
    result = {}
    for i, group in enumerate(ranked_groups):
        if topk is not None and i >= topk:
            break
        result[f"rank_{i+1}"] = group
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Rank patches into tie groups by reproduction fixes and regression passed (occurrence saved, not used)."
    )
    ap.add_argument("--summary_json", required=True, help="Path to the per-instance summary JSON.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, output top-k ranked groups.")
    ap.add_argument("--out", help="Write result JSON to a file instead of stdout.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for tie-breaking.")
    ap.add_argument("--pick_one_from_tie", action="store_true", help="Pick one patch randomly from top tie group.")

    args = ap.parse_args()

    summary = load_json(args.summary_json)
    ranked_groups = rank_patches(summary, seed=args.seed)
    winner = pick_top(ranked_groups, seed=args.seed, pick_one=args.pick_one_from_tie)

    output = {
        "instance_id": summary.get("instance_id"),
        "selected_patch" if args.pick_one_from_tie else "selected_patches": winner,
        "ranked_groups": group_with_rank_keys(ranked_groups, topk=args.topk if args.topk > 0 else None)
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
