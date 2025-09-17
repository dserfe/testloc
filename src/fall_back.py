import ast
import os
from typing import Dict, List, Tuple
from test_minimization import get_base_commit
import subprocess

from model_config import prompt_model


def prompt_model_to_select_test(
    tests, func, model_name
):
    prompt = f"""
You are an expert in test minimization. 
For functions `{func}`, below are the tests that may cover the functions.
Your task is to minimize the test suite, select top ten tests: 
###Function Body###
---BEGIN FUNCTION BODY---
{func}
---END FUNCTION BODY---
###Tests Covering the Function###
---BEGIN TESTS---
{tests}
---END TESTS---
You should follow the format below to provide your answer. Do not include any additional text or explanations.
keep tests in original json format. Here is an example:
---BEGIN SELECTED TESTS---
{{
file1: [
    test1, 
    test2
    ],
file2: [
    test3
  ]
}}
...
---END SELECTED TESTS---
"""
    print(f"Prompt for model {model_name}:\n{prompt}\n")
    response = prompt_model(model_name, prompt)
    print(f"Response\n{response}\n")
    if response:
        if (
            "---BEGIN SELECTED TESTS---" in response
            and "---END SELECTED TESTS---" in response
        ):
            selected_tests = (
                response.split("---BEGIN SELECTED TESTS---")[1]
                .split("---END SELECTED TESTS---")[0]
                .strip()
            )
            return ast.literal_eval(selected_tests)
    else:
        print(
            f"No response from model {model_name} for function {func}."
        )

def prep_repo(repo, base_commit, repo_base_dir = "temp_repos"):
    repo_dir = os.path.join(repo_base_dir, repo.split('/')[-1])

    if not os.path.exists(repo_dir):
        print(f"Cloning repository {repo}...")
        subprocess.run([
            "git", "clone", f"https://github.com/{repo}.git", repo_dir
        ], check=True)
    else:
        print(f"Repository {repo} already exists.")

    subprocess.run(["git", "fetch"], cwd=repo_dir, check=True)
    subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True)
    return repo_dir

def find_tests_calling_function(
    repo_dir: str, target_funcs: Dict[str, List[str]]
) -> List[Tuple[str, str, str, str]]:
    # repo_dir: Root of the codebase
    # target_funcs: Dict mapping source files to function names, e.g.,
    #         {"src/module.py": ["Class.method", "func", "Class"]}

    # List of (test_file_path, test_func_name, matched_name, func_source_file)
    results = []

    match_funcs = set()
    fallback_classes = set()

    for file_path, funcs in target_funcs.items():
        for f in funcs:
            parts = f.split(".")
            if len(parts) == 2 and parts[0][0].isupper():
                class_name, func_name = parts
                match_funcs.add((class_name, func_name, f, file_path))
                fallback_classes.add((class_name, f, file_path))
            elif len(parts) == 1:
                if parts[0][0].isupper():
                    fallback_classes.add((parts[0], f, file_path))  # just class
                else:
                    match_funcs.add(
                        (None, parts[0], f, file_path)
                    )  # top-level function

    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            full_path = os.path.join(root, file)

            if "test" not in os.path.basename(full_path).lower():
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=full_path)
            except Exception as e:
                print(f"Skipping {full_path} due to parse error: {e}")
                continue

            current_func = None
            import_aliases = {}  # e.g., {'u': 'astropy.units'}

            class TestVisitor(ast.NodeVisitor):
                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        if alias.asname:
                            import_aliases[alias.asname] = (
                                f"{node.module}.{alias.name}"
                                if node.module
                                else alias.name
                            )
                        else:
                            import_aliases[alias.name] = (
                                f"{node.module}.{alias.name}"
                                if node.module
                                else alias.name
                            )

                def visit_Import(self, node):
                    for alias in node.names:
                        if alias.asname:
                            import_aliases[alias.asname] = alias.name
                        else:
                            import_aliases[alias.name] = alias.name

                def visit_FunctionDef(self, node):
                    nonlocal current_func
                    current_func = node.name
                    self.generic_visit(node)

                def visit_Call(self, node):
                    nonlocal current_func

                    resolved_name = None

                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            alias = node.func.value.id
                            attr = node.func.attr
                            base = import_aliases.get(alias)
                            if base:
                                resolved_name = f"{base}.{attr}"
                            else:
                                resolved_name = f"{alias}.{attr}"
                        else:
                            resolved_name = node.func.attr
                    elif isinstance(node.func, ast.Name):
                        resolved_name = node.func.id

                    for class_name, func_name, full_name, func_file in match_funcs:
                        if resolved_name:
                            if class_name:
                                if resolved_name.endswith(f"{class_name}.{func_name}"):
                                    results.append(
                                        (full_path, current_func, full_name, func_file)
                                    )
                            else:
                                if resolved_name.endswith(func_name):
                                    results.append(
                                        (full_path, current_func, full_name, func_file)
                                    )

                    # Also detect instantiations like ClassName(...)
                    if isinstance(node.func, ast.Name):
                        class_name = node.func.id
                        for fallback_class, full_name, func_file in fallback_classes:
                            if class_name == fallback_class:
                                results.append(
                                    (full_path, current_func, full_name, func_file)
                                )

                    self.generic_visit(node)

                def visit_Attribute(self, node):
                    nonlocal current_func
                    if isinstance(node.value, ast.Name):
                        class_candidate = node.value.id
                        for fallback_class, full_name, func_file in fallback_classes:
                            if class_candidate == fallback_class:
                                results.append(
                                    (full_path, current_func, full_name, func_file)
                                )
                    self.generic_visit(node)

            visitor = TestVisitor()
            visitor.visit(tree)

    return results

# verified_claude_fallback = ['django__django-16315', 'astropy__astropy-13579', 'django__django-11728', 'sympy__sympy-23413', 'django__django-11099', 'sympy__sympy-17139', 'pylint-dev__pylint-6903', 'django__django-13809', 'django__django-15629', 'django__django-10097', 'pallets__flask-5014', 'django__django-11138', 'django__django-16255', 'pytest-dev__pytest-7982', 'django__django-7530', 'django__django-14376', 'pytest-dev__pytest-5262', 'django__django-16145', 'django__django-16082']
# SUSPICIOUS_JSON='more_related_suspicisous/agentless_verified_claude_filtered_suspicious.json'

# verified_gpt4o_fallback = ['sympy__sympy-13647', 'django__django-16527', 'sympy__sympy-23413', 'matplotlib__matplotlib-23314', 'sphinx-doc__sphinx-7440', 'django__django-11099', 'sympy__sympy-18189', 'django__django-14915', 'django__django-13809', 'django__django-15629', 'django__django-10097', 'pallets__flask-5014', 'django__django-11138', 'pylint-dev__pylint-6386', 'django__django-12304', 'django__django-7530', 'django__django-14376', 'pytest-dev__pytest-5262', 'django__django-16145']
# SUSPICIOUS_JSON='more_related_suspicisous/agentless_verified_gpt4o_filtered_suspicious.json'

# lite_gpt4o_fallback = ['pallets__flask-5063', 'sympy__sympy-15678', 'pytest-dev__pytest-5227', 'sympy__sympy-11400', 'pytest-dev__pytest-11148', 'pallets__flask-4045', 'pallets__flask-4992', 'sympy__sympy-21627', 'matplotlib__matplotlib-26011', 'django__django-11099', 'pytest-dev__pytest-5103', 'django__django-16255', 'sympy__sympy-22840', 'django__django-13448', 'pytest-dev__pytest-5221', 'sympy__sympy-22005', 'django__django-13660']
# SUSPICIOUS_JSON='more_related_suspicisous/agentless_lite_gpt4o_filtered_suspicious.json'

lite_claude_fallback = ['matplotlib__matplotlib-23562', 'pallets__flask-5063', 'django__django-14382', 'pytest-dev__pytest-5227', 'sympy__sympy-13437', 'scikit-learn__scikit-learn-15535', 'pytest-dev__pytest-11148', 'pallets__flask-4045', 'pallets__flask-4992', 'sympy__sympy-21627', 'sympy__sympy-16106', 'django__django-11099', 'sympy__sympy-17139', 'pytest-dev__pytest-5103', 'django__django-16255', 'sympy__sympy-12171', 'django__django-13230', 'django__django-12915', 'pytest-dev__pytest-5221', 'django__django-15781']
SUSPICIOUS_JSON='more_related_suspicisous/agentless_lite_claude_filtered_suspicious.json'

import json
from datasets import load_dataset

# read json to functions
with open(SUSPICIOUS_JSON, 'r') as f:
    suspicious_data = json.load(f)

datasetname = "princeton-nlp/SWE-bench_Lite"
dataset = load_dataset(datasetname)
dataset = dataset['test']
all = []
result_json = f"fall_back/fall_back_{SUSPICIOUS_JSON.split('/')[-1]}"

for instance_id in lite_claude_fallback:
    results = {}
    base_commit, repo_name = get_base_commit(instance_id, dataset)
    local_repo_path = prep_repo(repo_name, base_commit)
    normalized_funcs = suspicious_data.get(instance_id, [])
    print(f"Processing {instance_id} with {len(normalized_funcs)} suspicious functions")
    print(json.dumps(normalized_funcs, indent=2))
    if len(normalized_funcs) == 0:
        print("No suspicious functions found.")
        continue
    new_check = find_tests_calling_function(local_repo_path, normalized_funcs)
    print(json.dumps(new_check, indent=2))
    tests = {}
    for item in new_check:
        test_file, test_func, matched_name, func_source_file = item
        test_file = test_file.replace(local_repo_path + "/", "")
        if test_file not in tests:
            tests[test_file] = []
        if test_func not in tests[test_file] and test_func:
            tests[test_file].append(test_func)
            
    # check if total values of tests is larger than 20, if so prompt a model to select only 20
    if sum(len(v) for v in tests.values()) > 20:
        selected_tests = prompt_model_to_select_test(tests, normalized_funcs, model_name = "claude-3-5-sonnet-20241022")
        results = {
            "instance_id": instance_id,
            "file_function": selected_tests
        }
    else:
        results = {
            "instance_id": instance_id,
            "file_function": tests
        }

    print(json.dumps(results, indent=2))
    if len(tests) > 0:
        all.append(results)

# save all to json
with open(result_json, 'w') as f:
    json.dump(all, f, indent=2)