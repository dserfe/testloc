import ast
import difflib
import json
import os
import re
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset

from locate_tests import locate_tests

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


def transform_path_to_module(file_path: str) -> str:
    if "temp_repos2" in file_path:
        file_path = "/".join(file_path.split("/")[2:])
    if file_path.startswith(("a/", "b/")):
        file_path = file_path[2:]

    if file_path.endswith(".py"):
        file_path = file_path[:-3]
    return file_path.replace("/", ".")


def get_modified_files(instance_id):
    for item in ds:
        if item["instance_id"] != instance_id:
            continue
        patch = item.get("test_patch", "")
        modified_files = set()
        for line in patch.splitlines():
            if line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line[4:].strip()
                if path.startswith(("a/", "b/")):
                    # path = path[2:]
                    path = transform_path_to_module(line[4:].strip())
                modified_files.add(path)
    modified_files = list(modified_files)
    return modified_files


def check_hit(gr_test_modules, close_tests):
    for gr_test in gr_test_modules:
        for test in close_tests:
            if test.startswith(gr_test + ".") or gr_test == test:
                print(f"Hit found: {gr_test} in {test}")
                return True
    return False


def read_json_files(directory: str) -> List[Dict[str, Any]]:
    json_results = []
    directory_path = Path(directory)
    num = 0
    all_results = {}

    for file in directory_path.glob("*.json"):
        num += 1
        # if num == 10:
        #     exit(0)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_path = os.path.join(directory, file.name)
            print(file_path)
            abs_path = os.path.abspath(file_path)
            json_results.append((data, abs_path))
            if "instance_id" in data:
                instance_id = data["instance_id"]
            else:
                for key in data:
                    instane_id = key
                    data = data[key]

            closest_tests_dict = data["closest_tests"]
            potential_test_methods = data["potential_test_methods"]
            potential_test_files = data["potential_test_files"]
            normalized_funcs = data["normalized_funcs"]
            modified_test_files = get_modified_files(instance_id)
            if instance_id not in all_results:
                all_results[instance_id] = {
                    "tests_in_call_graph": closest_tests_dict,
                    "potential_test_methods": potential_test_methods,
                    "potential_test_files": potential_test_files,
                    "normalized_funcs": normalized_funcs,
                    "modified_test_files": modified_test_files,
                }

            """
            continue
            repo_path = f"temp_repos2/{instance_id.split('__')[0]}/"
            
            method_with_zero_tests = []
            add_test_files = []
            closest_tests = []
            
            for method in closest_tests_dict:
                tests = closest_tests_dict[method]
                if len(tests) == 0:
                    # print(f"No closest tests found for method {method} in instance {instance_id}")
                    method_with_zero_tests.append(method)
                else:
                    closest_tests.extend(tests)

            for method in method_with_zero_tests:
                for file in normalized_funcs:
                    if method in normalized_funcs[file]:
                        # print("potential test files:", potential_test_files[file])
                        transformed = [transform_path_to_module(file) for file in potential_test_files[file]]
                        # print("Transformed potential test files:", transformed)

            closest_tests.extend(add_test_files)
            closest_tests = list(set(closest_tests))
            fq_closest_tests = [to_pytest_qualified_name(test, repo_path) for test in closest_tests]
            print(f"Processing instance: {instance_id}, found {len(closest_tests)} closest tests")

            modified_test_files = get_modified_files(instance_id)
            filtered_map = select_top_similar_test_files(potential_test_files, top_num=15)
            all_test_files = []
            filtered_files = []
            for file in potential_test_files:
                stage1, stage2, stage3, combined = get_closest_test_files_with_staged_heuristics(file, potential_test_files[file], top_n=5)
                print(f"File: {file}, Stage 1: {len(stage1)}, Stage 2: {len(stage2)}, Stage 3: {len(stage3)}, Combined: {len(combined)}")
                print(f"Stage 1: {stage1}")
                print(f"Stage 2: {stage2}")
                print(f"Stage 3: {stage3}")
                print(f"Combined: {combined}")
                filtered_files.extend(combined)
                for m in potential_test_files[file]:
                    rel = transform_path_to_module(m)
                    all_test_files.append(rel)

            filtered_files = list(set(filtered_files))
            fq_filtered_files = [transform_path_to_module(file) for file in filtered_files]
            
            all_test_methods = []
            for file in potential_test_methods:
                for test_file in potential_test_methods[file]:
                    for test_method in potential_test_methods[file][test_file]:
                        all_test_methods.append(test_method.replace(repo_path, ""))

            fq_closest_tests.extend(all_test_methods)
            fq_closest_tests = list(set(fq_closest_tests))
            print(f"Total closest tests: {len(closest_tests)}")
            print(f"modified_test_files: {modified_test_files}")
            if_hit = check_hit(modified_test_files, fq_filtered_files)
            # for t in closest_tests:
            #     print(t)
            print(f"All closest tests: {len(fq_closest_tests)}")
            print(f"Filtered test files: {len(filtered_files)}")
            # print(fq_closest_tests)
            print(f"Hit: {if_hit}")
            # if len(fq_closest_tests) < 50:
            #     print(f"Running coverage for {instance_id} with {len(fq_closest_tests)} tests")
            #     run_coverage_docker(instance_id, fq_closest_tests)
            # else:
            #     print(f"Skipping coverage run for {instance_id} due to too many tests ({len(fq_closest_tests)})")
            # exit(0)
            """

    # save all_results to a single JSON file
    output_file = "verified_all_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    return json_results


import subprocess


def run_coverage_docker(instance_id, test_methods):
    print(
        f"Running coverage for instance {instance_id} with {len(test_methods)} test methods"
    )
    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "princeton-nlp/SWE-bench_Verified",
        "--predictions_path",
        "gold",
        "--max_workers",
        "3",
        "--run_id",
        "test",
        "--instance_ids",
        instance_id,
        "--test_methods",
        json.dumps(test_methods),  # ensure proper string escaping
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result


def to_pytest_qualified_name(name: str, repo_path: str) -> str:
    parts = name.strip().split(".")

    # Try longest prefix as path
    for i in range(len(parts), 0, -1):
        file_candidate = os.path.join(repo_path, *parts[:i]) + ".py"
        if os.path.isfile(file_candidate):
            rel_file_path = os.path.relpath(file_candidate, repo_path)
            obj_suffix = "::" + "::".join(parts[i:]) if i < len(parts) else ""
            return rel_file_path + obj_suffix

    # Fallback: treat whole thing as unresolved
    return name


def select_top_similar_test_files(suspicious_map, top_num=5):
    filtered_map = {}
    for suspicious_path, test_files in suspicious_map.items():
        if not test_files:
            continue
        top_tests = get_top_similar_tests([suspicious_path], test_files, top_num)
        filtered_map[suspicious_path] = top_tests.get(suspicious_path, [])
    return filtered_map


import difflib
from typing import List


def get_top_similar_tests(main_methods, test_methods, top_num):
    result = {}

    for main_method in main_methods:
        if not test_methods:
            return []

        similarities = []
        for test_method in test_methods:
            ratio = difflib.SequenceMatcher(None, main_method, test_method).ratio()
            similarities.append((test_method, ratio))

        similarities.sort(key=lambda x: x[1], reverse=True)

        k = max(1, top_num)
        top_tests = [method for method, _ in similarities[:k]]
        result[main_method] = top_tests
    return result


from pathlib import Path


def levenshtein_distance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, replace = previous[j] + 1, current[j - 1] + 1, previous[j - 1]
            if a[j - 1] != b[i - 1]:
                replace += 1
            current[j] = min(add, delete, replace)
    return current[n]


def rank_by_edit_distance_on_full_path(main_path_str, test_files):
    scored = []
    for tf in test_files:
        tf_path_str = str(Path(tf).with_suffix(""))  # remove .py for fair comparison
        dist = levenshtein_distance(main_path_str, tf_path_str)
        scored.append((dist, tf))
    scored.sort()  # lower distance is better
    return [tf for _, tf in scored]


def get_closest_test_files_with_staged_heuristics(main_file, test_files, top_n=10):
    main_path = Path(main_file)
    main_stem = main_path.stem
    main_path_str = str(main_path.with_suffix(""))  # full path without .py

    stage1_list = []
    variants = {
        f"test_{main_stem}.py",
        f"{main_stem}_test.py",
        f"test_{main_stem}s.py",
        f"{main_stem}s_test.py",
    }
    for tf in test_files:
        if Path(tf).name in variants:
            stage1_list.append(tf)

    if not stage1_list:
        ranked_stage1 = rank_by_edit_distance_on_name(main_stem, test_files)
        stage1_list = ranked_stage1[:top_n]

    main_module_parts = set(main_path.parts)
    stage2_candidates = []
    for tf in test_files:
        tf_path = Path(tf)
        tf_parts = set(tf_path.parts)
        if "tests" in tf_parts and len(main_module_parts & tf_parts) > 1:
            stage2_candidates.append(tf)

    stage2_list = rank_by_edit_distance_on_full_path(main_path_str, stage2_candidates)
    stage2_list = [t for t in stage2_list if t not in stage1_list]

    remaining_candidates = [
        t for t in test_files if t not in stage1_list and t not in stage2_list
    ]
    stage3_list = rank_by_edit_distance_on_name(main_stem, remaining_candidates)
    stage3_list = [
        t for t in stage3_list if t not in stage1_list and t not in stage2_list
    ]

    # Combine while preserving order and uniqueness
    combined_list = []
    # seen = set()
    # for lst in [stage1_list, stage2_list, stage3_list]:
    #     for item in lst:
    #         if item not in seen:
    #             combined_list.append(item)
    #             seen.add(item)
    #         if len(combined_list) >= top_n:
    #             break
    #     if len(combined_list) >= top_n:
    #         break
    # combine top 3 of each stage in to the combined list
    tok_k = 3
    combined_list.extend(stage1_list[:tok_k])
    combined_list.extend(stage2_list[:tok_k])
    combined_list.extend(stage3_list[:tok_k])

    return stage1_list, stage2_list, stage3_list, combined_list


def rank_by_edit_distance_on_name(main_stem, test_files):
    scored = []
    for tf in test_files:
        tf_name = (
            Path(tf).stem.replace("test_", "").replace("_test", "").replace(".py", "")
        )
        dist = levenshtein_distance(main_stem, tf_name)
        scored.append((dist, tf))
    scored.sort()  # lower distance is better
    return [tf for _, tf in scored]


if __name__ == "__main__":
    dir_path = "debugging_verified_logs9/princeton-nlp/SWE-bench_Verified/GCP_claude-3-7-sonnet/GCP_claude-3-7-sonnet_results/"
    json_data = read_json_files(dir_path)
    print(f"Loaded {len(json_data)} JSON files")
    # process_data(json_data)
