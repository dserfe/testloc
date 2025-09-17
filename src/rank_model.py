#!/usr/bin/env python3
import argparse, json, os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import urllib.request, urllib.error, urllib.parse, ssl
from datetime import datetime

hf_dataset = "princeton-nlp/SWE-bench_Verified"
hf_split = "test"

from datasets import load_dataset
ds = load_dataset(hf_dataset, split=hf_split)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def read_occurrence_from_patch_jsonl(patch_path: str) -> int:
    occ = 0
    try:
        for obj in iter_jsonl(patch_path):
            if isinstance(obj, dict):
                for k in ("occurrence", "occurrences", "votes"):
                    if k in obj:
                        try:
                            occ = max(occ, int(obj[k]))
                        except Exception:
                            pass
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return occ

def extract_patch_text_from_jsonl(patch_jsonl_path: str, max_chars: int = 8000) -> str:
    preferred = ("normalized_patch", "model_patch", "patch", "diff")
    try:
        for obj in iter_jsonl(patch_jsonl_path):
            if isinstance(obj, dict):
                for k in preferred:
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()[:max_chars]
                return (json.dumps(obj, indent=2, ensure_ascii=False)[:max_chars])
            else:
                return str(obj)[:max_chars]
    except Exception:
        return ""
    return ""

# --------------------------
# Aggregation & ranking
# --------------------------

def aggregate_repro(summary: dict) -> Tuple[Dict[str,int], int]:
    repro = summary.get("reproduction", {}) or {}
    flips: Dict[str, int] = defaultdict(int)
    total_considered = 0

    reg = summary.get("regression", {}) or {}
    for r in reg.get("results", []) or []:
        pf = r.get("patch_file")
        if pf:
            flips.setdefault(pf, 0)

    for tb in repro.values():
        if not isinstance(tb, dict):
            continue
        if not tb.get("repro_fails_on_clean", False):
            continue
        total_considered += 1
        for oc in (tb.get("outcomes", []) or []):
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
    s = set()
    reg = summary.get("regression", {}) or {}
    for r in reg.get("results", []) or []:
        if r.get("patch_file"): s.add(r["patch_file"])
    repro = summary.get("reproduction", {}) or {}
    for tb in repro.values():
        for oc in (tb.get("outcomes", []) or []):
            if oc.get("patch_file"): s.add(oc["patch_file"])
    return sorted(s)

def rank_patches(summary: dict) -> List[dict]:
    repro_flips, total_repro = aggregate_repro(summary)
    reg_pass = aggregate_regression(summary)
    patch_files = collect_patch_files(summary)

    ranked = []
    for pf in patch_files:
        ranked.append({
            "patch_file": pf,
            "repro_fixes": repro_flips.get(pf, 0),
            "repro_total_considered": total_repro,
            "regression_passed": reg_pass.get(pf, 0),
            "occurrence": read_occurrence_from_patch_jsonl(pf),
        })

    ranked.sort(key=lambda d: (-d["repro_fixes"],
                               -d["regression_passed"],
                               -d["occurrence"],
                                d["patch_file"]))
    return ranked

def top_ties(ranked: List[dict]) -> List[dict]:
    if not ranked:
        return []
    top = ranked[0]
    key = (top["repro_fixes"], top["regression_passed"])
    return [d for d in ranked if (d["repro_fixes"], d["regression_passed"]) == key]

# --------------------------
# HF issue fetch (problem_statement only)
# --------------------------


def get_issue_text_from_hf(instance_id: str,
                           hf_dataset: str,
                           hf_split: str) -> Optional[str]:

    if "instance_id" not in ds.column_names or "problem_statement" not in ds.column_names:
        return None

    for ex in ds:
        if ex.get("instance_id") == instance_id:
            ps = ex.get("problem_statement")
            if isinstance(ps, str) and ps.strip():
                return ps.strip()
            return None
    return None

# --------------------------
# Model call (OpenAI-compatible)
# --------------------------

def call_chat_api(messages: List[Dict[str, str]],
                  api_base: Optional[str],
                  api_key: Optional[str],
                  model: str,
                  max_tokens: int = 512,
                  temperature: float = 0.8,
                  timeout: int = 120) -> Optional[str]:
    if not api_base:
        return None
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        return resp_data["choices"][0]["message"]["content"]
    except Exception:
        return None

def choose_best_with_model(tied: List[dict],
                           issue_text: Optional[str],
                           max_chars_per_patch: int,
                           api_base: Optional[str],
                           api_key: Optional[str],
                           model: str,
                           timeout: int = 120,
                           log: Optional[dict] = None,
                           log_prompts: bool = True) -> Optional[dict]:
    if not tied or not api_base:
        return None

    blocks = []
    for i, d in enumerate(tied, 1):
        pf = d["patch_file"]
        snippet = extract_patch_text_from_jsonl(pf, max_chars=max_chars_per_patch)
        blocks.append(
            f"\n[{i}] patch_file: {pf}\n"
            "---- BEGIN PATCH ----\n"
            f"{snippet}\n"
            "---- END PATCH ----\n"
        )
    patch_blocks = "".join(blocks)

    system_prompt = (
        "Given an issue description and several candidate patches, "
        "select the single patch most likely to FIX the issue correctly.\n"
        "Respond ONLY:\n"
        "###Patch File Starts###<exact file name from candidates>###Patch File Ends###\n"
    )

    user_prompt = (
        f"ISSUE:\n-----\n{issue_text or ''}\n\n"
        f"CANDIDATE PATCHES (tied group size={len(tied)}):\n"
        f"{patch_blocks}\n\n"
        "Choose EXACTLY ONE patch_file from the candidates."
    )
    prompt = f"{system_prompt}\n{user_prompt}"

    messages = [
        {"role": "system", "content": "You are an expert Python developer assisting Automated Program Repair.\n"},
        {"role": "user", "content": prompt},
    ]

    if log is not None and log_prompts:
        log["model_prompt"] = {
            "system": system_prompt,
            "user": user_prompt[:200000]  # guard huge logs
        }
    print(user_prompt)
    content = call_chat_api(messages, api_base, api_key, model,
                            max_tokens=512, temperature=0.8, timeout=timeout)
    if log is not None:
        log["model_used"] = bool(content)
        if content is None:
            log["model_error"] = "No response / HTTP error"
        elif log.get("log_raw_response", False):
            log["model_raw_response"] = content

    if not content:
        return None

    # try:
    if True:
        print(content)
        patch_file = content.split("###Patch File Starts###")[1].split("###Patch File Ends###")[0].replace("\"", "").strip()
        print(patch_file)
        for d in tied:
            print("Checking patch:", d["patch_file"])
            if d["patch_file"] == patch_file:
                print("Here", d)
                return d
    # except Exception as e:
    #     if log is not None:
    #         log["model_parse_error"] = str(e)
    #     return None
    print("No matching patch found")
    return None

# --------------------------
# Logging
# --------------------------

def write_log(log_dir: str, instance_id: Optional[str], payload: dict):
    os.makedirs(log_dir, exist_ok=True)
    fname = f"{(instance_id or 'unknown').replace('/', '_')}.log.json"
    path = os.path.join(log_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Rank patches; if exact ties, read patch contents and ask a model to pick the most relevant to the issue (from HF dataset). Saves a per-instance log."
    )
    ap.add_argument("--summary_json", required=True, help="Path to the per-instance summary JSON.")
    ap.add_argument("--topk", type=int, default=0, help="If >0, also include top-k list in output.")
    ap.add_argument("--out", help="Write result JSON to a file instead of stdout.")

    # Model API
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"),
                    help="OpenAI-compatible API base (e.g., https://api.openai.com).")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                    help="API key.")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                    help="Model name (default: gpt-4o).")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout for model call.")
    ap.add_argument("--max-chars-per-patch", type=int, default=8000,
                    help="Max characters per patch snippet to show the model.")

    # HF dataset config
    ap.add_argument("--hf-dataset", default=os.environ.get("HF_DATASET", "princeton-nlp/SWE-bench_Verified"),
                    help="HF dataset name (default: princeton-nlp/SWE-bench_Verified).")
    ap.add_argument("--hf-split", default=os.environ.get("HF_SPLIT", "test"),
                    help="Dataset split (train/validation/test). Default: test.")

    # Logging
    ap.add_argument("--log-dir", default="logs_new/model_select",
                    help="Directory to write per-instance logs (default: logs/model_select).")
    ap.add_argument("--log-prompts", action="store_true",
                    help="Also save the exact prompts sent to the model (may be large).")
    ap.add_argument("--log-raw-response", action="store_true",
                    help="Also save the raw model response text.")

    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"Summary JSON exists at {args.out}.")
        return

    summary = load_json(args.summary_json)
    instance_id = summary.get("instance_id")

    ranked = rank_patches(summary)

    issue_text = None
    if instance_id:
        issue_text = get_issue_text_from_hf(instance_id, args.hf_dataset, args.hf_split)

    # Build base log payload
    run_log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "instance_id": instance_id,
        "summary_json": os.path.abspath(args.summary_json),
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "issue_text_present": bool(issue_text),
        "issue_text_chars": len(issue_text) if isinstance(issue_text, str) else 0,
        "ranking_count": len(ranked),
        "ranking_top": ranked[0] if ranked else None,
        "tie_group_size": 0,
        "model_endpoint": args.api_base,
        "model_name": args.model,
        "log_raw_response": args.log_raw_response,
        "used_model_for_tie": False,
        "winner_source": None,  # "model" | "deterministic" | "no_candidates"
    }

    ties = top_ties(ranked)
    run_log["tie_group_size"] = len(ties)

    if len(ties) > 1:
        run_log["used_model_for_tie"] = True
        run_log["tied_candidates"] = [d["patch_file"] for d in ties]

        winner = choose_best_with_model(
            ties,
            issue_text=issue_text,
            max_chars_per_patch=args.max_chars_per_patch,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            log=run_log,
            log_prompts=args.log_prompts
        )
        if not winner:
            winner = ties[0]
            run_log["winner_source"] = "deterministic"
        else:
            run_log["winner_source"] = "model"
    else:
        winner = ranked[0] if ranked else {}
        run_log["winner_source"] = "no_candidates" if not ranked else "deterministic"

    run_log["winner"] = winner.get("patch_file") if isinstance(winner, dict) else None
    log_path = write_log(args.log_dir, instance_id, run_log)

    output = {
        "instance_id": instance_id,
        "issue_text": issue_text,
        "selected_patch": winner,
        "ranked_patches": ranked[:args.topk] if args.topk and args.topk > 0 else ranked,
        "_log_path": log_path,
    }

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)

if __name__ == "__main__":
    main()
