import json
from itertools import chain

from datasets import load_dataset

from model_config import prompt_model
from prompts import trim_down_tests_prompt

dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
dataset = dataset["test"]


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def process_test_files(potential_test_files_json, model_name):
    all_results = {}

    for instance_id in potential_test_files_json:
        print(f"Processing instance: {instance_id}")
        content = potential_test_files_json[instance_id]
        potential_test_files = content["potential_test_files"]
        potential_test_methods = content["potential_test_methods"]
        tests_in_call_graph = content["tests_in_call_graph"]
        all_test_files = list(
            chain.from_iterable(
                potential_test_files[file] for file in potential_test_files
            )
        )
        all_test_methods = []
        for main_file in potential_test_methods:
            for test_file in potential_test_methods[main_file]:
                all_test_methods.extend(potential_test_methods[main_file][test_file])

        print(
            f"Instance {instance_id} has {len(all_test_files)} test files and {len(all_test_methods)} test methods."
        )
        filtered_test_files = {}
        for python_file in potential_test_files:
            top_k = 0.1 * len(potential_test_files[python_file])
            top_k = max(1, int(top_k))  # Ensure at least one test
            print(f"Processing {python_file} with top_k={top_k} tests.")
            prompt = trim_down_tests_prompt(
                python_file, potential_test_files[python_file], top_k
            )
            response = prompt_model(model_name, prompt)
            print(response)
            test_files = parse_response(response)
            print(f"Filtered test files for {python_file}:\n{test_files}")
            filtered_test_files[python_file] = test_files
        if instance_id not in all_results:
            all_results[instance_id] = content
        all_results[instance_id]["filtered_test_files"] = filtered_test_files
        # break

    return all_results


def parse_response(response):
    test_files = []
    if (
        "---BEGIN RELEVANT TESTS---" in response
        and "---END RELEVANT TESTS---" in response
    ):
        test_files_section = response.split("---BEGIN RELEVANT TESTS---")[1].split(
            "---END RELEVANT TESTS---"
        )[0]
        test_files = [
            line.strip() for line in test_files_section.splitlines() if line.strip()
        ]
    else:
        print("No relevant test files found in the response.")
    return test_files


def cal_hit(filtered_test_file_json):
    hit = 0
    total = 0
    for instance_id in filtered_test_file_json:
        content = filtered_test_file_json[instance_id]
        potential_test_files = content["potential_test_files"]
        filtered_test_files = content["filtered_test_files"]
        modified_test_files = content["modified_test_files"]
        tests_in_call_graph = content["tests_in_call_graph"]
        found = []
        for python_file in filtered_test_files:
            for test_file in filtered_test_files[python_file]:
                if "temp_repos2" in test_file:
                    test_file = "/".join(test_file.split("/")[2:])
                if test_file in modified_test_files:
                    print(f"{instance_id} Hit: {test_file} in {modified_test_files}")
                    found.append(test_file)
        for main_method in tests_in_call_graph:
            for test_method in tests_in_call_graph[main_method]:
                for m in modified_test_files:
                    fq_m = m.replace(".py", "").replace("/", ".")
                    if test_method.startswith(fq_m):
                        print(
                            f"Call {instance_id} Hit: {test_method} in {modified_test_files}"
                        )
                        found.append(test_method)
                        # exit(0)
        if len(found) > 0:
            hit += 1
    print(hit)


def format_test(filtered_test_file_json):
    final_tests_results = {}
    nums = []
    hit_ins = []
    for instance_id in filtered_test_file_json:
        # print(f"Formatting tests for instance: {instance_id}")
        content = filtered_test_file_json[instance_id]
        potential_test_files = content["potential_test_files"]
        potential_test_methods = content["potential_test_methods"]
        filtered_test_files = content["filtered_test_files"]
        modified_test_files = content["modified_test_files"]
        tests_in_call_graph = content["tests_in_call_graph"]

        final_tests = []
        for python_file in filtered_test_files:
            test_files = filtered_test_files[python_file]
            for test_file in test_files:
                if test_file not in potential_test_files[python_file]:
                    # print(f"Warning: {test_file} not found in potential_test_files for {python_file}")
                    continue
                if test_file not in potential_test_methods[python_file]:
                    # print(f"Warning: {test_file} not found in potential_test_methods for {python_file}")
                    continue
                test_methods = potential_test_methods[python_file][test_file]
                for test_method in test_methods:
                    if "temp_repos2" in test_method:
                        repo_root = "/".join(test_method.split("/")[:2]) + "/"
                        test_method = test_method.replace(repo_root, "")
                    final_tests.append(test_method)
        for main_method in tests_in_call_graph:
            all_tests = tests_in_call_graph[main_method]
            hit_methods, repo_path, base_commit = check_test_coverage_in_patch(
                instance_id
            )
            all_tests = [
                to_pytest_qualified_name(test, repo_path) for test in all_tests
            ]
            final_tests.extend(all_tests)
        final_tests = list(set(final_tests))
        print(f"Instance {instance_id} has {len(final_tests)} final tests.")
        nums.append(len(final_tests))
        final_tests_results[instance_id] = final_tests

    print(f"Average number of final tests: {sum(nums) / len(nums)}")
    # print(f"Total instances with hits: {len(list(set(hit_ins)))}")
    return final_tests_results


import os

from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


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


def check_test_coverage_in_patch(instance_id):
    results = []
    hit_methods = []

    for item in ds:
        if item["instance_id"] != instance_id:
            continue
        repo_path = os.path.join("temp_repos2", item["repo"].split("/")[-1])
        base_commit = item["base_commit"]
        break

    return hit_methods, repo_path, base_commit


def get_issue_by_instance_id(instance_id):
    for item in dataset:
        if item["instance_id"] == instance_id:
            return item["problem_statement"] + "\n" + item["hints_text"]


def process_tests(potential_tests, model_name):
    all_results = {}
    # read all_results from JSON
    if os.path.exists("filtered_selected_tests.json"):
        with open("filtered_selected_tests.json", "r", encoding="utf-8") as f:
            all_results = json.load(f)
    for instance_id in potential_tests:
        try:
            if instance_id in all_results:
                print(f"Instance {instance_id} already processed. Skipping.")
                continue
            all_tests = potential_tests[instance_id]

            # check all_tests
            tests = [
                test
                for file_dict in all_tests.values()
                for tests in file_dict.values()
                for test in tests
            ]
            if len(tests) == 0:
                print(f"No tests found for instance {instance_id}. Skipping.")
                continue
            issue = get_issue_by_instance_id(instance_id)
            prompt = trim_down_tests_prompt(
                issue, all_tests, top_k=0.1 * len(all_tests)
            )
            response = prompt_model(model_name, prompt)
            print(f"Prompt for instance {instance_id}:\n{prompt}")
            print(f"Response for instance {instance_id}:\n{response}")
            tests = parse_response(response)
            print(f"Filtered tests for instance {instance_id}:\n{tests}")
            all_results[instance_id] = tests

            # save all_results to JSON
            with open("filtered_selected_tests.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)
        except Exception as e:
            print(f"Error processing instance {instance_id}: {e}")
            continue
    return all_results


if __name__ == "__main__":
    model_name = "GCP/claude-3-7-sonnet"
    # """
    potential_test_json = "selected_tests.json"
    potential_tests = read_json_file(potential_test_json)

    all_results = process_tests(potential_tests, model_name)

    # output_file = "trim_selected_tests.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, indent=4)
    # print(f"Filtered test files saved to {output_file}")
    # """

    # filtered_test_file = "filtered_test_files.json"
    # filtered_test_file_json = read_json_file(filtered_test_file)
    # final_tests_results = format_test(filtered_test_file_json)

    # output_file = "final_tests_results.json"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(final_tests_results, f, indent=4)
    # print(f"Final tests results saved to {output_file}")
    # cal_hit(filtered_test_file_json)
