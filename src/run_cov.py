import json
import os

from count_match import run_coverage_docker


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm


def get_cov(results, max_workers=3):

    result_dir = "logs/run_evaluation/test/gold"

    def log(msg):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {msg}", flush=True)

    def process_instance(instance):
        all_tests = results[instance]
        num_tests = len(all_tests)
        log(f"[START] {instance} - Total tests: {num_tests}")

        if num_tests == 0:
            log(f"[SKIP] {instance} - No tests to run, skipping.")
            return

        test_output_txt = f"{result_dir}/{instance}/test_output.txt"

        if os.path.exists(test_output_txt):
            log(f"[SKIP] {instance} - Test output exists: {test_output_txt}")
        else:
            run_coverage_docker(instance, all_tests)

        log(f"[DONE] {instance}")

    # Sort instances by number of tests (ascending)
    instances = sorted(results.keys(), key=lambda inst: len(results[inst]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        total=len(instances), desc="Processing Instances", ncols=80
    ) as pbar:
        futures = {
            executor.submit(process_instance, instance): instance
            for instance in instances
        }

        for future in as_completed(futures):
            instance = futures[future]
            try:
                future.result()
            except Exception as e:
                log(f"[ERROR] {instance} - Exception: {e}")
            finally:
                pbar.update(1)


if __name__ == "__main__":
    json_path = "final_tests_results.json"
    results = read_json_file(json_path)
    get_cov(results)
