import argparse
import os
import sys

import localization
from utils import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            An agentic approach.
            """,
    )
    parser.add_argument(
        "--instance_id",
        dest="instance_id",
        required=False,
        help="An instance to inspect, if not provided, it will run on all instances.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="GCP/claude-3-7-sonnet",  # "mistralai/mistral-large", #required = True
        help="LLM model to run.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset to run on.",
    )
    parser.add_argument(
        "--log_dir",
        dest="log_dir",
        default="ablation_debugging_verified_logs",
        help="Directory to store logs.",
    )
    if not os.path.exists(parser.parse_args().log_dir):
        os.makedirs(parser.parse_args().log_dir)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    instance_id = args.instance_id
    model = args.model
    dataset_name = args.dataset
    log_dir = args.log_dir

    dataset = get_dataset(dataset_name)
    log_dir = f'{log_dir}/{dataset_name}/{model.replace("/", "_")}'

    localization.main(model, dataset_name, dataset, log_dir, instance_id)
