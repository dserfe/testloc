"""
A script to find related call paths of a method in a repo.
Four modes:
  1. all_paths: Find all paths between two functions.
  2. forward: Find all forward paths from a start function.
  3. related: Find all related paths (both callers and callees) to a target function.
"""

import json
import sys
from collections import defaultdict


def find_all_paths(graph, start, end, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]
    visited.add(start)

    if start == end:
        return [path]

    if start not in graph:
        return []

    paths = []
    for node in graph[start]:
        if node not in visited:
            new_paths = find_all_paths(graph, node, end, path, visited.copy())
            for p in new_paths:
                paths.append(p)
    return paths


def get_all_forward_callees(call_paths_json, start):
    """
    Given a call graph and a start method, return a set of all forward callees.
    """
    graph = load_graph_from_json(call_paths_json)
    paths = explore_all_forward_paths(graph, start)
    callees = set()
    for path in paths:
        for method in path[1:]:  # exclude the start method itself
            callees.add(method)
    return list(callees)


def explore_all_forward_paths(graph, start, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]
    visited.add(start)

    paths = [path]

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            paths += explore_all_forward_paths(graph, neighbor, path, visited.copy())

    return paths


def build_reverse_graph(graph):
    print("building reverse graph...")
    reverse_graph = defaultdict(set)
    for caller, callees in graph.items():
        for callee in callees:
            reverse_graph[callee].add(caller)
    return reverse_graph


def explore_all_backward_paths(graph, start, path=None, visited=None):
    print(f"Exploring backward paths from {start}...")
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]
    visited.add(start)

    paths = [path]

    for predecessor in graph.get(start, []):
        if predecessor not in visited:
            paths += explore_all_backward_paths(
                graph, predecessor, path, visited.copy()
            )

    return paths


def load_graph_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    # replace '.self.' with '.' in all keys and values
    new_graph = {}
    for key, values in graph.items():
        new_key = key.replace(".self.", ".")
        new_values = [v.replace(".self.", ".") for v in values]
        new_graph[new_key] = new_values
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_graph, f, indent=2)

    return new_graph


def load_graph_from_log(log_path):
    graph = defaultdict(set)
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "->" in line:
                caller, callee = line.strip().split(" -> ")
                graph[caller].add(callee)
    return graph


def find_lca(paths):
    if not paths:
        return []

    # Reverse paths so they go from root to target
    reversed_paths = [list(reversed(p)) for p in paths]
    min_len = min(len(p) for p in reversed_paths)

    lca = []
    for i in range(min_len):
        current = reversed_paths[0][i]
        if all(p[i] == current for p in reversed_paths):
            lca.append(current)
        else:
            break

    if lca:
        return lca

    # No common prefix found; find the least set of mutual ancestors
    all_ancestors = set()
    for path in reversed_paths:
        all_ancestors.update(path)

    # remove any ancestor that has another ancestor as descendant
    redundant = set()
    for a in all_ancestors:
        for b in all_ancestors:
            if a != b and a in reversed_paths_dict.get(b, []):
                redundant.add(a)

    return sorted(all_ancestors - redundant)


def clean_tests(call_paths_json, logger=None):
    """
    Remove test methods from the call graph.
    """
    with open(call_paths_json, "r", encoding="utf-8") as f:
        graph = json.load(f)

    cleaned_graph = {}
    for key, values in graph.items():
        if "test_" not in key.lower() and ".tests." not in key.lower():
            cleaned_graph[key] = [
                v
                for v in values
                if not v.split(".")[-1].lower().startswith("test_")
                and ".tests." not in v.lower()
            ]

    non_test_json_path = call_paths_json.replace(".json", "_non_test.json")
    with open(non_test_json_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_graph, f, indent=2)
    logger.info(f"Non-test json saved to {non_test_json_path}")
    return non_test_json_path


import concurrent.futures
import threading


def related_mode(
    call_paths_json, target, exclude_tests=False, logger=None, _result_holder=None
):
    if _result_holder is None:
        _result_holder = {}

    logger.info(f"Finding all related paths for '{target}' at {call_paths_json}'...")
    print(f"Finding all related paths for '{target}' at {call_paths_json}'...")

    if exclude_tests:
        print("Excluding test methods from the call graph...")
        non_test_json_path = clean_tests(call_paths_json, logger)
        call_paths_json = non_test_json_path

    if ".json" in call_paths_json:
        graph = load_graph_from_json(call_paths_json)
    else:
        graph = load_graph_from_log(call_paths_json)

    print("Getting all related paths...")

    reverse_graph = build_reverse_graph(graph)
    backward_paths = explore_all_backward_paths(reverse_graph, target)
    _result_holder["backward_paths"] = backward_paths

    if not backward_paths:
        logger.info(f"No related paths found for '{target}'.")
        _result_holder["lca"] = []
        return []
    else:
        logger.info(f"\nAll related paths for:\n  {target}\n")
        logger.info("Caller paths:")
        for i, path in enumerate(backward_paths, 1):
            logger.info(f"Path {i}:")
            for step in reversed(path):
                logger.info(f"  → {step}")
            logger.info("\n")

        global reversed_paths_dict
        reversed_paths_dict = {path[-1]: set(path) for path in backward_paths if path}

        lca = find_lca(backward_paths)
        _result_holder["lca"] = lca

        if lca:
            logger.info("Lowest Common Ancestor(s) of all caller paths:")
            for node in lca:
                logger.info(f"  → {node}")
            logger.info("\n")

    if not exclude_tests:
        lca_test_methods = [method for method in lca if "test" in method.lower()]
        return lca_test_methods
    else:
        return lca


def run_with_timeout_relatedmode(timeout_seconds, *args, **kwargs):
    print(f"Running related mode with a timeout of {timeout_seconds} seconds...")
    result_holder = {}

    def target():
        try:
            related_mode(*args, _result_holder=result_holder, **kwargs)
        except Exception as e:
            result_holder["error"] = str(e)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(
            f"Timeout reached ({timeout_seconds} seconds), returning partial result..."
        )
        thread.join(0)  # Optional: ensure thread cleanup

    if "lca" in result_holder:
        if kwargs.get("exclude_tests", False):
            return result_holder["lca"]
        else:
            return [m for m in result_holder.get("lca", []) if "test" in m.lower()]
    else:
        return []


# result = run_with_timeout_relatedmode(10, call_paths_json, target, exclude_tests=False, logger=my_logger)


def main():
    if len(sys.argv) not in [4, 5]:
        print("Usage:")
        print("  To find all paths between two functions:")
        print("    python find_paths.py all_paths <log_file> <start_func> <end_func>")
        print("  To find all forward paths from a start function:")
        print("    python find_paths.py forward <log_file> <start_func>")
        print("  To find all related paths (both callers and callees):")
        print("    python find_paths.py related <log_file> <target_func>")
        sys.exit(1)

    mode = sys.argv[1]
    log_file = sys.argv[2]
    graph = load_graph_from_log(log_file)

    if mode == "all_paths":
        start = sys.argv[3]
        end = sys.argv[4]
        paths = find_all_paths(graph, start, end)
        if not paths:
            print(f"No paths found from '{start}' to '{end}'.")
        else:
            print(f"\nAll paths from:\n  {start}\nto:\n  {end}\n")
            for i, path in enumerate(paths, 1):
                print(f"Path {i}:")
                for step in path:
                    print("  →", step)
                print()

    elif mode == "forward":
        start = sys.argv[3]
        paths = explore_all_forward_paths(graph, start)
        if not paths:
            print(f"No forward paths found from '{start}'.")
        else:
            print(f"\nAll forward paths from:\n  {start}\n")
            for i, path in enumerate(paths, 1):
                print(f"Path {i}:")
                for step in path:
                    print("  →", step)
                print()

    elif mode == "related":
        target = sys.argv[3]
        related_mode(log_file, target)

    else:
        print("Invalid mode. Use 'all_paths', 'forward', or 'related'.")


if __name__ == "__main__":
    main()
